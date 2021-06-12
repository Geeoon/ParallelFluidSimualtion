#include <iostream>
#include <vector>
#include <array>
#include <SFML/Graphics.hpp>
#include <amp.h>
#include <amp_math.h>

void diffuse(unsigned int xDim, unsigned int yDim, concurrency::array_view<double, 3> &x, concurrency::array_view<double, 3> &x0, double dt) {
	double a = dt * 0.01 * xDim * yDim;
	for (unsigned int k = 0; k < 20; k++) {
		for (unsigned int i = 1; i < xDim - 1; i++) {
			for (unsigned int j = 1; j < yDim - 1; j++) {
				x(i, j, 2) = (x0(i, j, 2) + a * (x(i - 1, j, 2) + x(i + 1, j, 2) + x(i, j - 1, 2) + x(i, j + 1, 2))) / (1 + 4 * a);
			}
		}
	}
}

int main() {
	const unsigned int xDim = 200;
	const unsigned int yDim = 200;
	const unsigned int size = xDim * yDim * 4;
	sf::Clock clock;
	double dt = 0;
	sf::RenderWindow window;
	window.create(sf::VideoMode{ xDim, yDim }, "Fluid Simulation", sf::Style::Close);
	window.setFramerateLimit(0);
	clock.restart();

	double* averageArray = new double[xDim * yDim * 3];
	for (int i = 0; i < xDim * yDim * 3; i++) {
		averageArray[i] = 0;
	}
	concurrency::array_view<double, 3> averages(xDim, yDim, 3, averageArray);  // stored as such: (x, y, [x velocity, y velocity, density])

	double* densityArray = new double[xDim * yDim];
	for (int i = 0; i < xDim * yDim; i++) {
		densityArray[i] = 0;
	}
	concurrency::array_view<double, 2> densitySources(xDim, yDim, densityArray);
	densitySources(xDim / 2, yDim / 2) = 10000;

	unsigned int* pixelArray = new unsigned int[size];
	for (int i = 0; i < size; i++) {
		pixelArray[i] = 0;
	}
	concurrency::array_view<unsigned int, 3> pixels(xDim, yDim, 4, pixelArray);  // must be stored in in array_view to be used in amp restricted lambda; stored as such: (x, y, [red, blue, green])
	
	sf::Uint8* pixelUINT = new sf::Uint8[size];  // used to convert unsigned int arry to sf::Uint8 arry
	while (window.isOpen()) {
		// logic and parallel processing
		// http://graphics.cs.cmu.edu/nsp/course/15-464/Spring11/papers/StamFluidforGames.pdf
		concurrency::array_view<double, 3> initialAverages = averages;
		double a = dt * 0.0001 * xDim * yDim;
		concurrency::parallel_for_each(averages.extent,  // update density from density source
			[=](concurrency::index<3> idx) restrict(amp) {  // "shader"
			int x = idx[0];
			int y = idx[1];
			int value = idx[2];  // property
			if (value == 2) {
				averages(x, y, value) += densitySources(x, y) * dt;
			}
			for (int i = 0; i < 20; i++) {
				averages(x, y, 2) = (initialAverages(x, y, 2) + a * (averages(x - 1, y, 2) + averages(x + 1, y, 2) + averages(x, y - 1, 2) + averages(x, y + 1, 2))) / (1 + 4 * a);
			}
		});

		concurrency::parallel_for_each(pixels.extent,
			[=](concurrency::index<3> idx) restrict(amp) {  // "shader"
			int x = idx[0];
			int y = idx[1];
			int channel = idx[2];  // RGBA
			if (channel == 3) {
				pixels(x, y, channel) = 255;  // alpha channel should always be 255
			} else {
				unsigned int pressure{ static_cast<unsigned int>(averages(x, y, 2)) };
				if (pressure > 255) {  // clip the pressure so it can be stored in a uint8
					pressure = 255;
				}
				pixels(x, y, channel) = pressure;  // draw the pressure
			}
			
		});
		// synchronize array_view and vector
		pixels.synchronize();
		averages.synchronize();



		// SFML stuff below; do not change
 
		// checking window events
		sf::Event e;
		while (window.pollEvent(e)) {
			if (e.type == sf::Event::Closed) {
				window.close();
			}
		}

		// creating image to manipulate pixels
		sf::Image image;
		
		// this is done instead of a reinterpret_cast, becuase the memory size of each element is different: 4 bytes (unsigned int) vs. 1 byte (sf::Uint8)
		for (int i = 0; i < size; i++) {
			pixelUINT[i] = pixelArray[i];  
		}

		image.create(xDim, yDim, pixelUINT);

		// loading image into texture to be drawn
		sf::Texture t;
		t.loadFromImage(image);
		
		window.clear();  // clear window to begin drawing

		// draw below 
		
		window.draw(sf::Sprite(t));
		
		// ^^^^^ draw above ^^^^^
		
		window.display();  // display image
		dt = clock.getElapsedTime().asSeconds();
		clock.restart();

		// ^^^^^ SFML stuff above; do not change ^^^^^

	}
	delete[] averageArray;
	delete[] densityArray;
	delete[] pixelArray;
	delete[] pixelUINT;
	return 0;
}