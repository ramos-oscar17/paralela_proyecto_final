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

    // Guardar resultados
    std::cout << "\nGuardando imágenes en disco...\n";
    savePPM("mandelbrot_paralelo_8k.ppm", image);
    savePPM("mandelbrot_blur_paralelo_8k.ppm", blurred_image);
    std::cout << "¡Proceso terminado con éxito!\n";

    return 0;
}