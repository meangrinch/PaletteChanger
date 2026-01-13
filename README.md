# PaletteChanger

A simple program that changes an images's color palette based on a provided image.

![Screenshot](docs/images/example_1.png)

![Screenshot](docs/images/example_2.png)

## Features

- Select an image file (various formats supported).
- Input an image as the color palette (auto limits to 256 colors).
- Adjust the number of colors to use from the inputted color palette.
- Enable "Enhanced contrast" to better preserve lines and features.
- Enable "Linearized colors" to reduce banding and noise (may also reduce overall brightness).
- Colors are mapped using the CIEDE2000 algorithm.

## Notes

Works better with actual color palettes as opposed to random images.
