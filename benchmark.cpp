#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <iomanip>
#include <string>
#include <omp.h>

const int WIDTH = 7680;
const int HEIGHT = 4320;
const int MAX_ITER = 256;

struct Pixel { uint8_t r, g, b; };

// El bucle ahora usa schedule(runtime) para aceptar configuraciones al vuelo
void generateMandelbrot(std::vector<Pixel>& image) {
    #pragma omp parallel for schedule(runtime)
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

// Función auxiliar para aislar la medición de tiempo
double measureTime(std::vector<Pixel>& image, omp_sched_t sched, int chunk) {
    omp_set_schedule(sched, chunk);
    
    auto start = std::chrono::high_resolution_clock::now();
    generateMandelbrot(image);
    auto end = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration<double>(end - start).count();
}

int main() {
    std::vector<Pixel> image(WIDTH * HEIGHT);
    std::vector<int> chunk_sizes = {1, 4, 16, 64, 256, 1024};
    
    int max_threads = omp_get_max_threads();
    std::cout << "--- Benchmark OpenMP Mandelbrot (8K) ---\n";
    std::cout << "Hilos disponibles: " << max_threads << "\n\n";
    
    // Configuración de la tabla de salida
    std::cout << std::left << std::setw(15) << "Planificador" 
              << std::setw(15) << "Chunk Size" 
              << "Tiempo (segundos)\n";
    std::cout << std::string(50, '-') << "\n";

    // 1. Evaluación: Static (Por defecto, divide HEIGHT / Hilos)
    // El chunk size '0' le indica a OpenMP que use el bloque por defecto
    double time_static = measureTime(image, omp_sched_static, 0);
    std::cout << std::left << std::setw(15) << "Static" 
              << std::setw(15) << "Auto" 
              << time_static << " s\n";
              
    std::cout << std::string(50, '-') << "\n";

    // 2. Evaluación: Dynamic con distintos tamaños de bloque
    for (int chunk : chunk_sizes) {
        double time = measureTime(image, omp_sched_dynamic, chunk);
        std::cout << std::left << std::setw(15) << "Dynamic" 
                  << std::setw(15) << chunk 
                  << time << " s\n";
    }
    
    std::cout << std::string(50, '-') << "\n";

    // 3. Evaluación: Guided con distintos tamaños de bloque mínimo
    // En guided, el chunk especifica el tamaño *mínimo* del bloque a asignar
    for (int chunk : chunk_sizes) {
        double time = measureTime(image, omp_sched_guided, chunk);
        std::cout << std::left << std::setw(15) << "Guided" 
                  << std::setw(15) << chunk 
                  << time << " s\n";
    }

    std::cout << "\nBenchmark finalizado.\n";
    return 0;
}