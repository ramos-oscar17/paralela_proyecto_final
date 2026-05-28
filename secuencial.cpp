#include <iostream>
#include <vector>
#include <fstream>
#include <complex>
#include <chrono>
#include <cmath>

// Configuración de la imagen a resolución 8K
const int WIDTH = 7680;
const int HEIGHT = 4320;
const int MAX_ITER = 256;

// Estructura para representar un píxel RGB
struct Pixel {
    uint8_t r, g, b;
};

// ==========================================
// Tarea A: Generación del Conjunto de Mandelbrot
// ==========================================
void generateMandelbrot(std::vector<Pixel>& image) {
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            // Mapear coordenadas del píxel al plano complejo
            double real = (x - WIDTH / 2.0) * 4.0 / WIDTH;
            double imag = (y - HEIGHT / 2.0) * 4.0 / WIDTH;
            std::complex<double> c(real, imag);
            std::complex<double> z(0, 0);
            
            int iter = 0;
            while (std::abs(z) < 2.0 && iter < MAX_ITER) {
                z = z * z + c;
                iter++;
            }
            
            // Coloreado simple basado en el número de iteraciones
            int idx = y * WIDTH + x;
            if (iter == MAX_ITER) {
                image[idx] = {0, 0, 0}; // Interior del conjunto (negro)
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
// Tarea B: Filtro de Convolución (Desenfoque Gaussiano)
// ==========================================
void applyGaussianFilter(const std::vector<Pixel>& input, std::vector<Pixel>& output) {
    const int RADIUS = 5; // Kernel de 11x11 (Pesado)
    const double SIGMA = 2.0;
    const int KERNEL_SIZE = RADIUS * 2 + 1;
    
    // Generar el kernel Gaussiano dinámicamente
    std::vector<double> kernel(KERNEL_SIZE * KERNEL_SIZE);
    double sum = 0.0;
    for (int ky = -RADIUS; ky <= RADIUS; ++ky) {
        for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
            double weight = std::exp(-(kx * kx + ky * ky) / (2 * SIGMA * SIGMA));
            kernel[(ky + RADIUS) * KERNEL_SIZE + (kx + RADIUS)] = weight;
            sum += weight;
        }
    }
    
    // Normalizar el kernel
    for (double& w : kernel) w /= sum;

    // Aplicar la convolución
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double r = 0, g = 0, b = 0;
            
            for (int ky = -RADIUS; ky <= RADIUS; ++ky) {
                for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
                    int ny = y + ky;
                    int nx = x + kx;
                    
                    // Manejo de bordes (clamp)
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

// Función auxiliar para guardar la imagen en formato PPM
void savePPM(const std::string& filename, const std::vector<Pixel>& image) {
    std::ofstream ofs(filename, std::ios::binary);
    ofs << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(image.data()), image.size() * sizeof(Pixel));
    ofs.close();
}

int main() {
    std::vector<Pixel> image(WIDTH * HEIGHT);
    std::vector<Pixel> blurred_image(WIDTH * HEIGHT);

    std::cout << "Iniciando procesamiento secuencial a " << WIDTH << "x" << HEIGHT << "...\n\n";

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
    std::cout << "Tarea B (Convolución 11x11) completada en: " << time_b.count() << " segundos.\n";

    // Guardar resultados
    std::cout << "\nGuardando imágenes en disco...\n";
    savePPM("mandelbrot_8k.ppm", image);
    savePPM("mandelbrot_blur_8k.ppm", blurred_image);
    std::cout << "¡Proceso terminado!\n";

    return 0;
}