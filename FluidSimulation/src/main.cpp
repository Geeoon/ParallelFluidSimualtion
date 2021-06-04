#include <iostream>
#include <array>
#include <SFML/Graphics.hpp>
#include <amp.h>
#include <amp_math.h>

int main() {
	const unsigned int xDim = 500;
	const unsigned int yDim = 500;
	const unsigned int size = xDim * yDim * 3;

	sf::RenderWindow window;
	window.create(sf::VideoMode{ xDim, yDim }, "Fluid Simulation", sf::Style::Close);

	while (window.isOpen()) {
		// logic and parallel processing
		unsigned int* pixelArray = new unsigned int[size];
		concurrency::array_view<unsigned int, 3> pixels(xDim, yDim, 3, pixelArray);

		concurrency::parallel_for_each(pixels.extent,
			[=](concurrency::index<3> idx) restrict(amp) {
			int x = idx[0];
			int y = idx[1];
			int channel = idx[2];
			pixels(x, y, channel) = (x + y) * 255 / (xDim + yDim);
		});

		pixels.synchronize();
		
		// checking window events
		sf::Event e;
		while (window.pollEvent(e)) {
			if (e.type == sf::Event::Closed) {
				window.close();
			}
		}

		// creating image to manipulate pixels
		sf::Image image;
		image.create(xDim, yDim, sf::Color::Black);

		// adding data from array to image
		for (int x = 0; x < xDim; x++) {
			for (int y = 0; y < yDim; y++) {
				image.setPixel(x, y, sf::Color(pixels(x, y, 0), pixels(x, y, 1), pixels(x, y, 2)));
			}
		}

		delete[] pixelArray;  // done with array

		// loading image into texture to be drawn
		sf::Texture t;
		t.loadFromImage(image);
		
		window.clear();  // clear window to begin drawing

		// draw below 
		
		window.draw(sf::Sprite(t));
		
		// ^^^^^ draw above ^^^^^
		
		window.display();  // display image
	}
	return 0;
}