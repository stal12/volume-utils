// Copyright(c) 2017 Federico Bolelli
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met :
// 
// *Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and / or other materials provided with the distribution.
// 
// * Neither the name of "OpenCV_Project_Generator" nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <random>

#include <opencv2/opencv.hpp>

#include "volume_util.h"
#include "file_manager.h"

using namespace std;
using namespace cv;
using namespace filesystem;

template <typename T>
struct interval {
	T begin;
	T end;
	T step;
};

//int find_threshold(const cv::String &filename) {
//	std::vector<cv::Mat1b> planes;
//	path vol_path(filename);
//	path files_path = vol_path / path("files.txt");
//	std::vector<std::string> planenames;
//
//	{
//		std::ifstream is(files_path.string());				// text mode shoud translate \r\n into \n only 
//		if (!is) {
//			return -1;
//		}
//
//		std::string cur_filename;
//		while (std::getline(is, cur_filename)) {
//			planenames.push_back(cur_filename);
//		}
//	}
//
//	for (unsigned int plane = 0; plane < planenames.size(); plane++) {
//		Mat1b tmp = imread((vol_path / path(planenames[plane])).string(), IMREAD_GRAYSCALE);
//		if (tmp.empty()) {
//			return -1;
//		}
//		planes.push_back(std::move(tmp));
//	}
//
//	Mat1b long_img;
//	hconcat(planes, long_img);
//
//	int thresh = (int) threshold(long_img, long_img, 0, 255, THRESH_OTSU);
//
//	imwrite("long_img.png", long_img);
//	return thresh;
//}


template<class URNG>
bool generate_density_volumes(const cv::String &directory, const interval<unsigned int>& density,
	const std::vector<unsigned int>& sizes, unsigned int images_per_size, URNG& urng) {

	path directory_path(directory);
	if (!create_directories(directory_path)) {
		return false;
	}
	path files_path = directory_path / path("files.txt");
	path logfile_path = directory_path / path("log.txt");

	ostringstream max_number;
	max_number << images_per_size - 1;
	size_t number_width = max_number.str().length();

	ofstream file_list(files_path.string(), ios::binary);
	if (!file_list) {
		return false;
	}

	ofstream log(logfile_path.string(), ios::binary);
	if (!log) {
		return false;
	}

	for (unsigned int s = 0; s < sizes.size(); s++) {
		unsigned int size = sizes[s];
		for (unsigned int d_index = 0, d = density.begin; d <= density.end; d += density.step, d_index++) {

			discrete_distribution<int> dis({ 100.0 - d , 0.0 + d });

			for (unsigned int i = 0; i < images_per_size; i++) {
				
				int sz[3] = { (int) size, (int) size, (int) size };
				Mat vol(3, sz, CV_8UC1);
				unsigned int foreground = 0;

				for (unsigned int z = 0; z < size; z++) {
					for (unsigned int y = 0; y < size; y++) {
						uchar * const row = vol.ptr<uchar>(z, y);
						for (unsigned int x = 0; x < size; x++) {
							unsigned char random_value = dis(urng);
							row[x] = random_value;
							if (random_value > 0) {
								foreground++;
							}
						}
					}
				}

				std::ostringstream vol_name;
				if (sizes.size() > 1) {
					vol_name << s;
				}
				vol_name << d_index << i;

				volwrite((directory_path / path(vol_name.str())).string(), vol, {IMWRITE_PNG_COMPRESSION, 9, IMWRITE_PNG_BILEVEL, 1});
				file_list << vol_name.str() << '\n';

				double real_density = (foreground * 100.0) / (size * size * size);

				log << vol_name.str() << '\t' 
					<< "real density: " << setprecision(3) << fixed << real_density << '\t' 
					<< "density error: " << setprecision(3) << fixed << abs(real_density - d) << '\n';
				cout << vol_name.str() << '\t' 
					<< "real density: " << setprecision(3) << fixed << real_density << '\t'
					<< "density error: " << setprecision(3) << fixed << abs(real_density - d) << endl;
 			}
		}
	}

	return true;
}



template<class URNG>
bool generate_granularity_volumes(const cv::String &directory, const interval<unsigned int>& granularity, const interval<unsigned int>& density, unsigned int size, unsigned int images_per_set, URNG& urng) {

	path directory_path(directory);
	if (!create_directories(directory_path)) {
		return false;
	}
	path files_path = directory_path / path("files.txt");
	path logfile_path = directory_path / path("log.txt");

	ostringstream max_number;
	max_number << images_per_set - 1;
	size_t number_width = max_number.str().length();

	ofstream file_list(files_path.string(), ios::binary);
	if (!file_list) {
		return false;
	}

	ofstream log(logfile_path.string(), ios::binary);
	if (!log) {
		return false;
	}

	int sz[3] = { (int)size, (int)size, (int)size };
	Mat vol(3, sz, CV_8UC1);

	for (unsigned int d = density.begin; d <= density.end; d += density.step) {

		discrete_distribution<int> dis({ 100.0 - d , 0.0 + d });

		for (unsigned int g = granularity.begin; g <= granularity.end; g += granularity.step) {

			for (unsigned int i = 0; i < images_per_set; i++) {

				unsigned int foreground = 0;

				for (unsigned int z = 0; z < size; z+=g) {
					for (unsigned int y = 0; y < size; y+=g) {
						uchar * const row = vol.ptr<uchar>(z, y);
						for (unsigned int x = 0; x < size; x+=g) {
							unsigned char random_value = dis(urng);

							for (unsigned zb = 0; zb < g && zb + z < size; zb++) {
								for (unsigned yb = 0; yb < g && yb + y < size; yb++) {
									for (unsigned xb = 0; xb < g && xb + x < size; xb++) {

										vol.at<unsigned char>(z + zb, y + yb, x + xb) = random_value;
										if (random_value > 0) {
											foreground++;
										}

									}
								}
							}
						}
					}
				}

				std::ostringstream vol_name;
				vol_name << setw(2) << setfill('0') << g;
				vol_name << setw(3) << setfill('0') << d;
				vol_name << setw(number_width) << setfill('0') << i;

				volwrite((directory_path / path(vol_name.str())).string(), vol, { IMWRITE_PNG_COMPRESSION, 9, IMWRITE_PNG_BILEVEL, 1 });
				file_list << vol_name.str() << '\n';

				double real_density = (foreground * 100.0) / (size * size * size);

				log << vol_name.str() << '\t'
					<< "real density: " << setprecision(3) << fixed << real_density << '\t'
					<< "density error: " << setprecision(3) << fixed << abs(real_density - d) << '\n';
				cout << vol_name.str() << '\t'
					<< "real density: " << setprecision(3) << fixed << real_density << '\t'
					<< "density error: " << setprecision(3) << fixed << abs(real_density - d) << endl;
			}
		}
	}

	return true;
}




bool expand_volume(const std::string& src, const std::string& dst, int mul) {

	cv::Mat vol = volread(src);
	if (vol.data == 0) {
		return false;
	}

	cv::Mat big_vol;
	int sz[] = { vol.size[0] * mul, vol.size[1] * mul, vol.size[2] * mul };
	big_vol.create(3, sz, vol.type());

	for (int z = 0; z < vol.size[0]; z++) {
		for (int y = 0; y < vol.size[1]; y++) {
			for (int x = 0; x < vol.size[2]; x++) {				

				for (int bz = z * mul; bz < z*mul + mul; bz++) {
					for (int by = y * mul; by < y*mul + mul; by++) {
						for (int bx = x * mul; bx < x*mul + mul; bx++) {

							for (int i = 0; i < vol.step[2]; i++) {
								*(big_vol.ptr<uchar>(bz, by, bx) + i) = *(vol.ptr<uchar>(z, y, x) + i);
							}

						}
					}
				}

			}
		}
	}

	return volwrite(dst, big_vol);

}




int main() {

	//if (false) {
	//	int thresh = find_threshold("D:\\Downloads\\cthead-8bit.tar\\cthead-8bit");

	//	cv::Mat volume = volread("D:\\Downloads\\cthead-8bit.tar\\cthead-8bit", IMREAD_GRAYSCALE, thresh);

	//	vector<int> params;
	//	params.push_back(IMWRITE_PNG_COMPRESSION);
	//	params.push_back(9);

	//	volwrite("out", volume, params);
	//}

	// Generate random volumes
	if (true) {
		vector<unsigned int> sizes;
		sizes.push_back(8);
		sizes.push_back(16);
		sizes.push_back(32);
		sizes.push_back(64);
		sizes.push_back(128);
		sizes.push_back(256);
		// sizes.push_back(512);

		//sizes.push_back(1024);
		//sizes.push_back(2048);

		interval<unsigned int> density1 = { 10, 90, 10 };
		interval<unsigned int> density2 = { 0, 100, 1 };
		interval<unsigned int> granularity = { 1, 16, 1 };

		mt19937 rng(0);

		if (false) {
			if (!generate_density_volumes("D:\\YACCLAB\\input\\random3D\\classical", density1, sizes, 5, rng))
				return EXIT_FAILURE;
		}

		if (true) {
			if (!generate_granularity_volumes("D:\\YACCLAB\\input\\random3D\\granularity", granularity, density2, 256, 3, rng))
				return EXIT_FAILURE;
		}
	}

	if (false) {
		expand_volume("D:\\YACCLAB_files\\input\\3dcheck\\1", "C:\\Users\\Stefano\\Desktop\\3dcheck", 16);
	}

    return EXIT_SUCCESS;
}
