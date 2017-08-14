/**
*Developed By Karan Bhagat
*February 2017
**/

#include <stdio.h>
#include <string>
#include <math.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//number of channels i.e. R G B
#define CHANNELS 3

//Cuda kernel for converting RGB image into a GreyScale image
__global__
void colorConvertToGrey(unsigned char *rgb, unsigned char *grey, int rows, int cols)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	//Compute for only those threads which map directly to 
	//image grid
	if (col < cols && row < rows)
	{
		int grey_offset = row * cols + col;
		int rgb_offset = grey_offset * CHANNELS;
	
    	unsigned char r = rgb[rgb_offset + 0];
	    unsigned char g = rgb[rgb_offset + 1];
	    unsigned char b = rgb[rgb_offset + 2];
	
	    grey[grey_offset] = r * 0.299f + g * 0.587f + b * 0.114f;
    }
}

size_t loadImageFile(unsigned char *grey_image, const std::string &input_file, int *rows, int *cols );

void outputImage(const std::string &output_file, unsigned char *grey_image, int rows, int cols);

unsigned char *h_rgb_image; //store image's rbg data

int main(int argc, char **argv) 
{
	std::string input_file;
	std::string output_file;

	//Check for the input file and output file names
	switch(argc) {
		case 3:
			input_file = std::string(argv[1]);
			output_file = std::string(argv[2]);
            break;
		default:
			std::cerr << "Usage: <executable> input_file output_file";
			exit(1);
	}
	
	unsigned char *d_rgb_image; //array for storing rgb data on device
	unsigned char *h_grey_image, *d_grey_image; //host and device's grey data array pointers
	int rows; //number of rows of pixels
	int cols; //number of columns of pixels
	
	//load image into an array and retrieve number of pixels
	const size_t total_pixels = loadImageFile(h_grey_image, input_file, &rows, &cols);

	//allocate memory of host's grey data array
	h_grey_image = (unsigned char *)malloc(sizeof(unsigned char*)* total_pixels);

	//allocate and initialize memory on device
	cudaMalloc(&d_rgb_image, sizeof(unsigned char) * total_pixels * CHANNELS);
	cudaMalloc(&d_grey_image, sizeof(unsigned char) * total_pixels);
	cudaMemset(d_grey_image, 0, sizeof(unsigned char) * total_pixels);
	
	//copy host rgb data array to device rgb data array
	cudaMemcpy(d_rgb_image, h_rgb_image, sizeof(unsigned char) * total_pixels * CHANNELS, cudaMemcpyHostToDevice);

	//define block and grid dimensions
	const dim3 dimGrid((int)ceil((cols)/16), (int)ceil((rows)/16));
	const dim3 dimBlock(16, 16);
	
	//execute cuda kernel
	colorConvertToGrey<<<dimGrid, dimBlock>>>(d_rgb_image, d_grey_image, rows, cols);

	//copy computed gray data array from device to host
	cudaMemcpy(h_grey_image, d_grey_image, sizeof(unsigned char) * total_pixels, cudaMemcpyDeviceToHost);

	//output the grayscale image
	outputImage(output_file, h_grey_image, rows, cols);
	cudaFree(d_rgb_image);
	cudaFree(d_grey_image);
	return 0;
}

//function for loading an image into rgb format unsigned char array
size_t loadImageFile(unsigned char *grey_image, const std::string &input_file, int *rows, int *cols) 
{
	cv::Mat img_data; //opencv Mat object

	//read image data into img_data Mat object
	img_data = cv::imread(input_file.c_str(), CV_LOAD_IMAGE_COLOR);
	if (img_data.empty()) 
	{
		std::cerr << "Unable to laod image file: " << input_file << std::endl;
	}
		
	*rows = img_data.rows;
	*cols = img_data.cols;

	//allocate memory for host rgb data array
	h_rgb_image = (unsigned char*) malloc(*rows * *cols * sizeof(unsigned char) * 3);
	unsigned char* rgb_image = (unsigned char*)img_data.data;

	//populate host's rgb data array
	int x = 0;
	for (x = 0; x < *rows * *cols * 3; x++)
	{
		h_rgb_image[x] = rgb_image[x];
	}
	
	size_t num_of_pixels = img_data.rows * img_data.cols;
	
	return num_of_pixels;
}

//function for writing gray data array to the image file
void outputImage(const std::string& output_file, unsigned char* grey_image, int rows, int cols)
{
	//serialize gray data array into opencv's Mat object
	cv::Mat greyData(rows, cols, CV_8UC1,(void *) grey_image);
	//write Mat object to file
	cv::imwrite(output_file.c_str(), greyData);
}
