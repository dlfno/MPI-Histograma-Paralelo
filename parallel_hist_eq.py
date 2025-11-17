import numpy as np
import cv2
from mpi4py import MPI
import sys

def worker_histogram_equalization(channel):
    """
    Realiza la igualación de histograma en un solo canal (array 2D).
    """
    # Se asume que el canal es un array 2D (H, W) de tipo uint8
    H, W = channel.shape
    total_pixels = H * W

    # 1. Calcular Histograma de 256 bins
    # MÁS RÁPIDO Y DIRECTO: Usar bincount para enteros uint8
    hist = np.bincount(channel.ravel(), minlength=256)
    
    # Asegurarnos de que no tenga más de 256 (por si acaso)
    if len(hist) > 256:
        hist = hist[:256]

    # 2. Calcular CDF (Cumulative Distribution Function)
    cdf = hist.cumsum()

    # 3. Encontrar CDF_min (el primer valor no cero del CDF)
    # Usamos un array enmascarado para encontrar fácilmente el mínimo no cero.
    cdf_m = np.ma.masked_equal(cdf, 0)
    
    # Si cdf_m.min() falla (ej. imagen negra), usamos 0
    try:
        cdf_min = cdf_m.min()
    except ValueError:
        cdf_min = 0 # El canal es completamente negro

    # 4. Calcular la LUT (Look-Up Table) según la fórmula
    # LUT[i] = round[ ((CDF[i] - CDF_min) / (TotalPixels - CDF_min)) * 255 ]
    
    denominador = (total_pixels - cdf_min)

    # Evitar división por cero si todos los píxeles son iguales
    if denominador == 0:
        lut = np.arange(256, dtype=np.uint8) # No hacer nada
    else:
        numerador = (cdf - cdf_min)
        lut = np.round((numerador / denominador) * 255)
        
        # Asegurarse de que los valores están en el rango [0, 255]
        lut = np.clip(lut, 0, 255)

    lut = lut.astype(np.uint8)

    # 5. Aplicar la LUT para obtener el canal ecualizado
    # C'(x, y) = LUT[C(x, y)]
    equalized_channel = lut[channel]
    
    # Calcular el histograma de la imagen procesada (para verificación)
    hist_eq = np.bincount(equalized_channel.ravel(), minlength=256)
    
    # Asegurarnos de que no tenga más de 256 (por si acaso)
    if len(hist_eq) > 256:
        hist_eq = hist_eq[:256]

    return equalized_channel, hist, hist_eq

def sequential_equalization(image):
    """Versión secuencial para benchmarking."""
    H, W, _ = image.shape
    img_eq = np.zeros_like(image)
    
    # Procesar B, G, R secuencialmente
    for i in range(3):
        img_eq[:, :, i], _, _ = worker_histogram_equalization(image[:, :, i])
        
    return img_eq

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Este diseño requiere exactamente 1 Master y 3 Workers
    if size != 4:
        if rank == 0:
            print("Error: Este script debe ejecutarse con 4 procesos (1 Master + 3 Workers).")
            print(f"Ejemplo: mpirun -n 4 python {sys.argv[0]}")
        comm.Abort(1)

    # Definir roles
    MASTER_RANK = 0
    WORKER_B_RANK = 1
    WORKER_G_RANK = 2
    WORKER_R_RANK = 3

    # --- Variables de Master ---
    image = None
    final_image = None
    seq_time = 0.0
    mpi_time = 0.0
    
    # Metadatos [Alto, Ancho]
    metadata = np.empty(2, dtype=int)

    # =============== FASE 1: MASTER (Carga y Tarea Secuencial) ===============
    if rank == MASTER_RANK:
        print("--- Iniciando Proceso de Ecualización de Histograma Paralelo ---")
        print(f"Procesos totales: {size} (1 Master, 3 Workers)")

        # --- Carga de Imagen Propia ---
        # Se espera el nombre de archivo como argumento
        if len(sys.argv) < 2:
            print("Error: Debes proporcionar la ruta de la imagen como argumento.")
            print(f"Uso: mpirun -n 4 python {sys.argv[0]} <ruta_a_tu_imagen>")
            comm.Abort(1) # Abortar todos los procesos MPI
        
        image_path = sys.argv[1]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image is None:
            print(f"Error: No se pudo cargar la imagen desde '{image_path}'.")
            print("Verifica la ruta y los permisos del archivo.")
            comm.Abort(1)

        # Obtener dimensiones de la imagen cargada
        H, W, channels = image.shape
        
        if channels != 3:
            print(f"Error: La imagen debe ser RGB (3 canales), pero tiene {channels}.")
            comm.Abort(1)
            
        metadata[0], metadata[1] = H, W
        print(f"Imagen '{image_path}' cargada: {W}x{H}")
        # --- Fin de Carga ---
        
        # Guardar original para verificación
        cv2.imwrite("imagen_original.png", image)
        print("Imagen original (copia) guardada en 'imagen_original.png'")

        # --- Benchmarking Secuencial ---
        print("Ejecutando versión secuencial para benchmarking (T_s)...")
        start_seq = MPI.Wtime()
        _ = sequential_equalization(image.copy()) # Ejecutar la versión secuencial
        end_seq = MPI.Wtime()
        seq_time = end_seq - start_seq
        print(f"Tiempo Secuencial (T_s): {seq_time:.6f} segundos")
        print("-" * 60)
        print("Iniciando ejecución paralela (T_p)...")
        
        # Iniciar temporizador MPI
        start_mpi = MPI.Wtime()

    # =============== Broadcast y Send ===============
    
    # 1. Master transmite [H, W] a todos los Workers
    comm.Bcast(metadata, root=MASTER_RANK)
    H, W = metadata[0], metadata[1]

    # 2. Workers preparan buffers de recepción
    if rank != MASTER_RANK:
        recv_buf = np.empty((H, W), dtype=np.uint8)
    
    # 3. Master envía canales (B, G, R) a los Workers (1, 2, 3)
    if rank == MASTER_RANK:
        # Extraer canales. Deben ser contiguos en memoria para MPI.
        channel_b = np.ascontiguousarray(image[:, :, 0])
        channel_g = np.ascontiguousarray(image[:, :, 1])
        channel_r = np.ascontiguousarray(image[:, :, 2])

        # Enviar (bloqueante) - CORREGIDO: MPI.BYTE
        comm.Send([channel_b, MPI.BYTE], dest=WORKER_B_RANK)
        comm.Send([channel_g, MPI.BYTE], dest=WORKER_G_RANK)
        comm.Send([channel_r, MPI.BYTE], dest=WORKER_R_RANK)
    
    # 4. Workers reciben su canal
    elif rank == WORKER_B_RANK:
        comm.Recv([recv_buf, MPI.BYTE], source=MASTER_RANK)
        print(f"[Rank {rank}]: Recibido canal Blue.")
    elif rank == WORKER_G_RANK:
        comm.Recv([recv_buf, MPI.BYTE], source=MASTER_RANK)
        print(f"[Rank {rank}]: Recibido canal Green.")
    elif rank == WORKER_R_RANK:
        comm.Recv([recv_buf, MPI.BYTE], source=MASTER_RANK)
        print(f"[Rank {rank}]: Recibido canal Red.")

    # =============== Workers ===============
    
    if rank in [WORKER_B_RANK, WORKER_G_RANK, WORKER_R_RANK]:
        print(f"[Rank {rank}]: Procesando histograma...")
        processed_channel, _, _ = worker_histogram_equalization(recv_buf)
        
        # Enviar resultado de vuelta al Master - CORREGIDO: MPI.BYTE
        comm.Send([processed_channel, MPI.BYTE], dest=MASTER_RANK)
        print(f"[Rank {rank}]: Canal procesado enviado.")

    # =============== FASE 4: MASTER (Recolección y Verificación) ===============

    if rank == MASTER_RANK:
        # Preparar buffers para recibir los canales procesados
        proc_b = np.empty((H, W), dtype=np.uint8)
        proc_g = np.empty((H, W), dtype=np.uint8)
        proc_r = np.empty((H, W), dtype=np.uint8)

        # Recibir (el orden no importa, pero las fuentes sí)
        # CORREGIDO: MPI.BYTE
        comm.Recv([proc_b, MPI.BYTE], source=WORKER_B_RANK)
        print(f"[Rank {rank}]: Recibido canal Blue procesado.")
        
        comm.Recv([proc_g, MPI.BYTE], source=WORKER_G_RANK)
        print(f"[Rank {rank}]: Recibido canal Green procesado.")
        
        comm.Recv([proc_r, MPI.BYTE], source=WORKER_R_RANK)
        print(f"[Rank {rank}]: Recibido canal Red procesado.")
        
        # Detener temporizador MPI
        end_mpi = MPI.Wtime()
        mpi_time = end_mpi - start_mpi
        print(f"Todos los canales recibidos. Tiempo Paralelo (T_p): {mpi_time:.6f} seg.")
        
        # Reconstruir la imagen final 
        final_image = cv2.merge([proc_b, proc_g, proc_r])
        
        # Guardar imagen con un nombre diferente para evitar sobreescribir la original
        cv2.imwrite("imagen_ecualizada_final.png", final_image)
        print("Imagen ecualizada guardada en 'imagen_ecualizada_final.png'")
        print("-" * 60)

        # --- Verificación de Salida ---
        print("--- Verificación de Salida (Master) ---")
        print(f"Img Original:  Shape={image.shape}, Dtype={image.dtype}")
        print(f"Img Procesada: Shape={final_image.shape}, Dtype={final_image.dtype}")
        
        channels_orig = cv2.split(image)
        channels_eq = cv2.split(final_image)
        names = ['Blue', 'Green', 'Red']

        for i in range(3):
            print(f"\n--- Canal {names[i]} ---")
            orig_ch = channels_orig[i]
            eq_ch = channels_eq[i]
            
            # Guardar canales individuales (Antes/Después)
            cv2.imwrite(f"channel_{names[i]}_0_original.png", orig_ch)
            cv2.imwrite(f"channel_{names[i]}_1_ecualizada.png", eq_ch)

            # Estadísticas
            print(f"  Stats (Original):  Mean={np.mean(orig_ch):.2f}, StdDev={np.std(orig_ch):.2f}, Sum={np.sum(orig_ch, dtype=np.int64)}")
            print(f"  Stats (Ecualizada): Mean={np.mean(eq_ch):.2f}, StdDev={np.std(eq_ch):.2f}, Sum={np.sum(eq_ch, dtype=np.int64)}")
            
            # Histogramas
            hist_orig = np.bincount(orig_ch.ravel(), minlength=256)
            hist_eq = np.bincount(eq_ch.ravel(), minlength=256)
            if len(hist_orig) > 256: hist_orig = hist_orig[:256]
            if len(hist_eq) > 256: hist_eq = hist_eq[:256]
            
            print(f"  Hist (Original):  Bins [0:5] = {hist_orig[:5]}")
            print(f"  Hist (Ecualizada): Bins [0:5] = {hist_eq[:5]}")
            print(f"  (La ecualización expande el histograma, como se esperaba)")
            
        print("-" * 60)

        # --- Análisis de Desempeño ---
        print("--- Análisis de Desempeño (Master) ---")
        print(f"Tiempo Secuencial (T_s): {seq_time:.6f} segundos")
        print(f"Tiempo Paralelo (T_p):   {mpi_time:.6f} segundos")

        if mpi_time > 0 and seq_time > 0:
            # Speedup (SU)
            speedup = seq_time / mpi_time
            print(f"Speedup (SU = T_s / T_p): {speedup:.4f}")
            
            # N = 3 (Número de procesos que realizan el cómputo)
            num_processors = 3
            
            # Eficiencia (E)
            efficiency = speedup / num_processors
            print(f"Eficiencia (E = SU / N, con N=3): {efficiency:.4f} (o {efficiency * 100:.2f}%)")
        else:
            print("No se pudo calcular el desempeño (tiempos inválidos).")
        
        print("-" * 60)

    # Barrera para asegurar que todos los procesos terminen limpiamente
    comm.Barrier()
    if rank == MASTER_RANK:
        print("Proceso completado exitosamente.")

if __name__ == "__main__":
    main()
