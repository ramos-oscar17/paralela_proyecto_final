#include <iostream>
#include <vector>
#include <fstream>
#include <complex>
#include <chrono>
#include <cmath>
#include <omp.h> // Biblioteca nativa de OpenMP

// Configuración de la imagen a resolución 8K
const int WIDTH = 7680;
const int HEIGHT = 4320;
const int MAX_ITER = 256;

struct Pixel {
    uint8_t r, g, b;
};

// ==========================================
// Tarea A: Generación del Conjunto de Mandelbrot (Paralelizada)
// ==========================================
void generateMandelbrot(std::vector<Pixel>& image) {
    // Se utiliza schedule(dynamic) porque las filas centrales requieren drásticamente 
    // más iteraciones que las exteriores. Un tamaño de bloque (chunk) de 16 ayuda 
    // a balancear la carga reduciendo el overhead de sincronización.

    //despues de hacer las pruebas, el ganador es dynamic, chunk size 16
    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double real = (x - WIDTH / 2.0) * 4.0 / WIDTH;
            double imag = (y - HEIGHT / 2.0) * 4.0 / WIDTH;
            std::complex<double> c(real, imag);
            std::complex<double> z(0, 0);
            
            int iter = 0;
            while (std::abs(z) < 2.0 && iter < MAX_ITER) {
                z = z * z + c;
                iter++;
            }
            
            int idx = y * WIDTH + x;
            if (iter == MAX_ITER) {
                image[idx] = {0, 0, 0};
            } else {
                image[idx] = {
                    static_cast<uint8_t>(iter % 256),
                    static_cast<uint8_t>((iter * 5) % 256),
                    static_cast<uint8_t>((iter * 13) % 256)
                };
            }
        }
    }
}

// ==========================================
// Tarea B: Filtro de Convolución (Paralelizada)
// ==========================================
void applyGaussianFilter(const std::vector<Pixel>& input, std::vector<Pixel>& output) {
    const int RADIUS = 5; // Kernel de 11x11
    const double SIGMA = 2.0;
    const int KERNEL_SIZE = RADIUS * 2 + 1;
    
    std::vector<double> kernel(KERNEL_SIZE * KERNEL_SIZE);
    double sum = 0.0;
    for (int ky = -RADIUS; ky <= RADIUS; ++ky) {
        for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
            double weight = std::exp(-(kx * kx + ky * ky) / (2 * SIGMA * SIGMA));
            kernel[(ky + RADIUS) * KERNEL_SIZE + (kx + RADIUS)] = weight;
            sum += weight;
        }
    }
    
    for (double& w : kernel) w /= sum;

    // Se utiliza schedule(static) porque el costo computacional por píxel y por fila 
    // es completamente uniforme (un kernel de 11x11 constante). Esto elimina 
    // cualquier overhead de asignación dinámica en tiempo de ejecución.
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double r = 0, g = 0, b = 0;
            
            for (int ky = -RADIUS; ky <= RADIUS; ++ky) {
                for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
                    int ny = y + ky;
                    int nx = x + kx;
                    
                    ny = std::max(0, std::min(HEIGHT - 1, ny));
                    nx = std::max(0, std::min(WIDTH - 1, nx));
                    
                    double weight = kernel[(ky + RADIUS) * KERNEL_SIZE + (kx + RADIUS)];
                    int pixel_idx = ny * WIDTH + nx;
                    
                    r += input[pixel_idx].r * weight;
                    g += input[pixel_idx].g * weight;
                    b += input[pixel_idx].b * weight;
                }
            }
            
            output[y * WIDTH + x] = {
                static_cast<uint8_t>(std::min(255.0, std::max(0.0, r))),
                static_cast<uint8_t>(std::min(255.0, std::max(0.0, g))),
                static_cast<uint8_t>(std::min(255.0, std::max(0.0, b)))
            };
        }
    }
}
#include <algorithm> // Para std::sort
#include <map>

// ==========================================
// Tarea C: Cálculo de Histograma de Colores (Paralelizado)
// ==========================================
std::vector<int> calculateColorHistogram(const std::vector<Pixel>& image) {
    // El tamaño exacto para cubrir todas las combinaciones RGB
    const int NUM_COLORS = 256 * 256 * 256; 
    
    // El histograma global que retornaremos
    std::vector<int> global_histogram(NUM_COLORS, 0);
    
    // Obtenemos cuántos hilos van a participar
    int max_threads = omp_get_max_threads();
    
    // Matriz 2D: Un histograma completo (64 MB) para cada hilo
    // Consumo total de memoria: max_threads * 64 MB
    std::vector<std::vector<int>> local_histograms(max_threads, std::vector<int>(NUM_COLORS, 0));

    // Fase 1: Conteo Local
    #pragma omp parallel
    {
        int tid = omp_get_thread_num(); // Identificador del hilo actual
        
        // Cada hilo procesa un bloque de píxeles y escribe SOLO en su histograma privado
        #pragma omp for schedule(static)
        for (size_t i = 0; i < image.size(); ++i) {
            const Pixel& p = image[i];
            // Codificamos el RGB en un único índice entero (0 a 16,777,215)
            int color_idx = (p.r << 16) | (p.g << 8) | p.b;
            local_histograms[tid][color_idx]++;
        }

        // Fase 2: Reducción Paralela
        // Ahora los hilos se dividen el trabajo de sumar los sub-histogramas
        #pragma omp for schedule(static)
        for (int i = 0; i < NUM_COLORS; ++i) {
            int sum = 0;
            for (int t = 0; t < max_threads; ++t) {
                sum += local_histograms[t][i];
            }
            global_histogram[i] = sum;
        }
    }

    return global_histogram;
}

// Función para mostrar los colores más frecuentes
void printTopColors(const std::vector<int>& histogram, int top_n = 10) {
    // Usamos un vector de pares (Frecuencia, Indice de Color)
    std::vector<std::pair<int, int>> non_zero_colors;
    
    for (int i = 0; i < histogram.size(); ++i) {
        if (histogram[i] > 0) {
            non_zero_colors.push_back({histogram[i], i});
        }
    }

    // Ordenar de mayor a menor frecuencia
    std::sort(non_zero_colors.rbegin(), non_zero_colors.rend());

    std::cout << "\n--- Top " << top_n << " Colores Más Frecuentes ---\n";
    std::cout << "Color (R, G, B) \t | Cantidad de Píxeles\n";
    std::cout << "-------------------------|--------------------\n";
    
    int limit = std::min(top_n, static_cast<int>(non_zero_colors.size()));
    for (int i = 0; i < limit; ++i) {
        int count = non_zero_colors[i].first;
        int color_idx = non_zero_colors[i].second;
        
        // Decodificar el índice a RGB usando operaciones de bits
        int r = (color_idx >> 16) & 0xFF;
        int g = (color_idx >> 8) & 0xFF;
        int b = color_idx & 0xFF;

        std::cout << "(" << r << ", " << g << ", " << b << ")\t | " << count << " píxeles\n";
    }
    std::cout << "Total de colores únicos encontrados: " << non_zero_colors.size() << "\n";
}

void savePPM(const std::string& filename, const std::vector<Pixel>& image) {
    std::ofstream ofs(filename, std::ios::binary);
    ofs << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(image.data()), image.size() * sizeof(Pixel));
    ofs.close();
}

int main() {
    std::vector<Pixel> image(WIDTH * HEIGHT);
    std::vector<Pixel> blurred_image(WIDTH * HEIGHT);

    // Informar sobre el entorno de ejecución paralelo
    int max_threads = omp_get_max_threads();
    std::cout << "Iniciando procesamiento en paralelo utilizando hasta " << max_threads << " hilos...\n\n";

    // ==========================================
    // EJECUCIÓN DE TAREAS
    // ==========================================

    // Medir Tarea A
    auto start_a = std::chrono::high_resolution_clock::now();
    generateMandelbrot(image);
    auto end_a = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_a = end_a - start_a;
    std::cout << "Tarea A (Mandelbrot) completada en: " << time_a.count() << " segundos.\n";

    // Medir Tarea B
    auto start_b = std::chrono::high_resolution_clock::now();
    applyGaussianFilter(image, blurred_image);
    auto end_b = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_b = end_b - start_b;
    std::cout << "Tarea B (Convolución) completada en: " << time_b.count() << " segundos.\n";

    // Medir Tarea C (Histograma)
    std::cout << "\nCalculando histograma de la imagen con desenfoque...\n";
    auto start_c = std::chrono::high_resolution_clock::now();
    std::vector<int> final_histogram = calculateColorHistogram(blurred_image);
    auto end_c = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_c = end_c - start_c;
    std::cout << "Tarea C (Histograma) completada en: " << time_c.count() << " segundos.\n";

    // ==========================================
    // RESULTADOS Y ESCRITURA EN DISCO
    // ==========================================

    // Imprimir el Top 10 de colores
    printTopColors(final_histogram, 10);

    // Guardar resultados
    std::cout << "\nGuardando imágenes en disco...\n";
    savePPM("mandelbrot_paralelo_8k.ppm", image);
    savePPM("mandelbrot_blur_paralelo_8k.ppm", blurred_image);
    
    std::cout << "¡Proceso terminado con éxito!\n";

    return 0;
}