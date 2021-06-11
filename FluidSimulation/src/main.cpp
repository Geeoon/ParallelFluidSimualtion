#include <iostream>
#include <stdint.h>
#include <vector>
#include <array>
#include <SFML/Graphics.hpp>
#include <amp.h>
#include <amp_math.h>

int main() {
	const unsigned int xDim = 500;
	const unsigned int yDim = 500;
	const unsigned int size = xDim * yDim * 4;
	sf::Clock clock;
	double dt = 0;
	sf::RenderWindow window;
	window.create(sf::VideoMode{ xDim, yDim }, "Fluid Simulation", sf::Style::Close);
	window.setFramerateLimit(0);
	clock.restart();
	while (window.isOpen()) {
		// logic and parallel processing
		//std::vector<unsigned int> pixelVector;  // vector of pixels to be stored and then drawn
		//pixelVector.resize(size);
		unsigned int *pixelArray = new unsigned int[size];
		concurrency::array_view<unsigned int, 3> pixels(xDim, yDim, 4, pixelArray);  // must be stored in in array_view to be used in amp restricted lambda; stored as such: (x, y, [red, blue, green])

		double* averageArray = new double[xDim * yDim * 3];
		concurrency::array_view<double, 3> averages(xDim, yDim, 3, averageArray);  // stored as such: (x, y, [x velocity, y velocity, pressure])
		
		// http://graphics.cs.cmu.edu/nsp/course/15-464/Spring11/papers/StamFluidforGames.pdf
		concurrency::parallel_for_each(pixels.extent,
			[=](concurrency::index<3> idx) restrict(amp) {  // "shader"
			int x = idx[0];
			int y = idx[1];
			int channel = idx[2];  // RGB
			if (channel == 3) {
				pixels(x, y, channel) = 255;
			} else {
				pixels(x, y, channel) = 255 * dt;
			}
		});
		// synchronize array_view and vector
		pixels.synchronize();
		averages.synchronize();

		// checking window events
		sf::Event e;
		while (window.pollEvent(e)) {
			if (e.type == sf::Event::Closed) {
				window.close();
			}
		}

		// creating image to manipulate pixels
		sf::Image image;
		
		sf::Uint8* pixelUINT = new sf::Uint8[size];  // used to convert unsigned int arry to sf::Uint8 arry
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
		delete[] pixelArray;
		delete[] averageArray;
		delete[] pixelUINT;
	}
	return 0;
}