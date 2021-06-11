#include <iostream>
#include <vector>
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
		std::vector<unsigned int> pixelVector;  // vector of pixels to be stored and then drawn
		pixelVector.resize(size);
		concurrency::array_view<unsigned int, 3> pixels(xDim, yDim, 3, pixelVector);  // must be stored in in array_view to be used in amp restricted lambda; stored as such: (x, y, [red, blue, green])

		std::vector<double> averageVector;  // vector of average parts of screen, it is the same as the pixels, but it is used for storing average data as a double
		averageVector.resize(size);
		concurrency::array_view<double, 3> averages(xDim, yDim, 3, averageVector);  // stored as such: (x, y, [x velocity, y velocity, pressure])
		
		// http://graphics.cs.cmu.edu/nsp/course/15-464/Spring11/papers/StamFluidforGames.pdf
		concurrency::parallel_for_each(pixels.extent,
			[=](concurrency::index<3> idx) restrict(amp) {  // "shader"
			int x = idx[0];
			int y = idx[1];
			int channel = idx[2];  // RGB
			pixels(x, y, channel) = 255;
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
		image.create(xDim, yDim, sf::Color::Black);

		// adding data from array to image
		for (int x = 0; x < xDim; x++) {
			for (int y = 0; y < yDim; y++) {
				image.setPixel(x, y, sf::Color(pixels(x, y, 0), pixels(x, y, 1), pixels(x, y, 2)));
			}
		}

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