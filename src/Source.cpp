#include "CImg.h"
#include <iostream>
#include <complex>
#include <vector>
#include <omp.h>
#include <algorithm>

typedef double el_type;
typedef std::vector<std::complex<el_type> > t_complex_vector;
const double Pi = 3.14159265359;

template <typename T>
inline T m_reverse(T a, int bit_len)
{
	T res = 0;
	for (int i = 0; i < bit_len; ++i)
	{
		bool bit = (a >> i) % 2;
		res |= bit << (bit_len - i - 1);
	}
	return res;
}

inline int my_log_2(int a)
{
	int res = 0;
	int tmp = 1;
	while (tmp != a)
	{
		tmp *= 2;
		res++;
	}
	return res;
}

void m_FFT_vectorized(el_type *src_real, el_type *src_im, el_type *res_real, el_type *res_im, const size_t size, bool mthread_param)
{
	size_t global_subsequence_size = 1;
	const size_t bit_length = my_log_2(size);
	const int iterations = my_log_2(size);
	if (src_real != res_real)
		res_real = src_real;
	if (src_im != res_im)
		res_im = src_im;
#pragma omp parallel for schedule(static) if(mthread_param == 1)
	for (size_t i = 1; i < size - 1; ++i)
	{
		size_t j = m_reverse(i, bit_length);
		if (j <= i) continue;
		std::swap(res_real[i], res_real[j]);
		std::swap(res_im[i], res_im[j]);
	}

	size_t subsequence_size = global_subsequence_size;
	for (size_t i = 0; i < iterations; ++i)
	{
#pragma omp parallel for schedule(static) if(mthread_param == 1)
		for (int j = 0; j < size / (subsequence_size * 2); ++j)
		{
#pragma omp simd
			for (int t = 0; t < subsequence_size; ++t)
			{
				size_t t_adress = j * subsequence_size * 2 + t;
				el_type temp_cos = cos(Pi / subsequence_size * t);
				el_type temp_sin = -sin(Pi / subsequence_size * t);
				el_type temp_first = (temp_cos * res_real[t_adress + subsequence_size] - temp_sin * res_im[t_adress + subsequence_size]);
				el_type temp_second = (temp_sin * res_real[t_adress + subsequence_size] + temp_cos * res_im[t_adress + subsequence_size]);
				el_type temp_real_t = res_real[t_adress] + temp_first;
				el_type temp_imag_t = res_im[t_adress] + temp_second;
				el_type temp_real_ss_plus_t = res_real[t_adress] - temp_first;
				el_type temp_imag_ss_plus_t = res_im[t_adress] - temp_second;
				res_real[t_adress] = temp_real_t;
				res_im[t_adress] = temp_imag_t;
				res_real[t_adress + subsequence_size] = temp_real_ss_plus_t;
				res_im[t_adress + subsequence_size] = temp_imag_ss_plus_t;
			}
		}
		subsequence_size *= 2;
	}
}

void m_FFT_reversed(el_type *src_real, el_type *src_im, const size_t size, bool mthread_param)
{
	m_FFT_vectorized(src_im, src_real, src_im, src_real, size, mthread_param);
	std::swap(src_im, src_real);
	for (size_t i = 0; i < size; ++i)
	{
		src_real[i] /= size;
		src_im[i] /= size;
	}
}

void m_2D_fft(el_type *src_real, el_type *src_imag, el_type *res_real, el_type *res_imag, const size_t n, const size_t m, bool mthread_param)
{
	el_type *temp_real = new el_type[m];
	el_type *temp_imag = new el_type[m];
#pragma omp parallel for if (mthread_param == 1)
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			temp_real[j] = src_real[i*m + j];
			temp_imag[j] = src_imag[i*m + j];
		}
		m_FFT_vectorized(temp_real, temp_imag, temp_real, temp_imag, m, 1);
		for (int j = 0; j < m; ++j)
		{
			res_real[i*m + j] = temp_real[j];
			res_imag[i*m + j] = temp_imag[j];
		}
	}
	delete[] temp_real;
	delete[] temp_imag;
	temp_real = new el_type[n];
	temp_imag = new el_type[n];
#pragma omp parallel for if (mthread_param == 1)
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			temp_real[j] = res_real[i + j * m];
			temp_imag[j] = res_imag[i + j * m];
		}
		m_FFT_vectorized(temp_real, temp_imag, temp_real, temp_imag, n, 1);
		for (int j = 0; j < n; ++j)
		{
			res_real[i + j * m] = temp_real[j];
			res_imag[i + j * m] = temp_imag[j];
		}
	}
}

void m_2D_fft_reversed(el_type *src_real, el_type *src_imag, el_type *res_real, el_type *res_imag, const size_t n, const size_t m, bool mthread_param)
{
	el_type *temp_real = new el_type[n];
	el_type *temp_imag = new el_type[n];
#pragma omp parallel for if (mthread_param == 1)
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			temp_real[j] = res_real[i + j * m];
			temp_imag[j] = res_imag[i + j * m];
		}
		m_FFT_reversed(temp_real, temp_imag, n, 1);
		for (int j = 0; j < n; ++j)
		{
			res_real[i + j * m] = temp_real[j];
			res_imag[i + j * m] = temp_imag[j];
		}
	}
	delete[] temp_real;
	delete[] temp_imag;
	temp_real = new el_type[m];
	temp_imag = new el_type[m];
#pragma omp parallel for if (mthread_param == 1)
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			temp_real[j] = src_real[i*m + j];
			temp_imag[j] = src_imag[i*m + j];
		}
		m_FFT_reversed(temp_real, temp_imag, m, 1);
		for (int j = 0; j < m; ++j)
		{
			res_real[i*m + j] = temp_real[j];
			res_imag[i*m + j] = temp_imag[j];
		}
	}
}

void normalize_data(el_type* src, size_t size)// NEEDS FIX
{
	el_type* x_min = std::min_element(src, src + size - 1);
	el_type* x_max = std::max_element(src, src + size - 1);
	el_type dif = *x_max - *x_min;
	for (size_t i = 0; i < size; ++i)
	{
		src[i] = (src[i] - *x_min) / dif * 255;
	}
}

void shift_data(el_type* src, size_t size)
{
	for (size_t i = 0; i < size; ++i)
	{
		for (size_t j = 0; j < size / 2; ++j)
		{
			std::swap(src[i*size + j], src[i*size + size / 2 + j]);
		}
	}
	for (size_t i = 0; i < size; ++i)
	{
		for (size_t j = 0; j < size / 2; ++j)
		{
			std::swap(src[i + j * size], src[i + (size / 2 + j) * size]);
		}
	}
}

void apply_low_pass_filter(el_type* real, el_type* imag, size_t size, size_t radius_in_pixels)
{
	size_t center = size / 2;
	for (size_t i = 0; i < size*size; ++i)
	{
		size_t x = i % size;
		size_t y = i / size;
		if ((x - center)*(x - center) + (y - center)*(y - center) > radius_in_pixels*radius_in_pixels)
		{
			real[i] = 0;
			imag[i] = 0;
		}
	}
}

int main(int argc, char* argv[])
{
	cimg_library::CImg<unsigned char> image("PATH HERE"), two_dim_fft_res(image.width(), image.height(), 1, image.spectrum(), 0), two_dim_fft_res_cut(image.width(), image.height(), 1, image.spectrum(), 0),
		inverse_two_dim_fft(image.width(), image.height(), 1, image.spectrum(), 0);
	std::vector<el_type*> channels(image.spectrum());
	std::vector<el_type*> channels_imag(image.spectrum());
	std::vector<el_type*> magnitudes(image.spectrum());
	//forward fft
	for (size_t i = 0; i < channels.size(); ++i)
	{
		channels[i] = new el_type[image.width() * image.height()];
		channels_imag[i] = new el_type[image.width() * image.height()];
		magnitudes[i] = new el_type[image.width() * image.height()];
		for (size_t j = 0; j < image.width() * image.height(); ++j)
		{
			channels[i][j] = static_cast<el_type>(image(j % image.width(), j / image.height(), 0, i));
			channels_imag[i][j] = 0.;
		}
		m_2D_fft(channels[i], channels_imag[i], channels[i], channels_imag[i], image.height(), image.width(), 1);
		for (size_t j = 0; j < image.width() * image.height(); ++j)
		{
			magnitudes[i][j] = sqrt(channels[i][j] * channels[i][j] + channels_imag[i][j] * channels_imag[i][j]);
		}
		normalize_data(magnitudes[i], image.width());
		shift_data(magnitudes[i], image.width());
		for (size_t j = 0; j < image.width() * image.height(); ++j)
		{
			two_dim_fft_res(j % image.width(), j / image.height(), 0, i) = magnitudes[i][j];
		}
		shift_data(channels[i], image.width());
		shift_data(channels_imag[i], image.width());
		apply_low_pass_filter(channels[i], channels_imag[i], image.width(), 300);
		for (size_t j = 0; j < image.width() * image.height(); ++j)
		{
			magnitudes[i][j] = sqrt(channels[i][j] * channels[i][j] + channels_imag[i][j] * channels_imag[i][j]);
			two_dim_fft_res_cut(j % image.width(), j / image.height(), 0, i) = magnitudes[i][j];
		}
		shift_data(channels[i], image.width());
		shift_data(channels_imag[i], image.width());
		m_2D_fft_reversed(channels[i], channels_imag[i], channels[i], channels_imag[i], image.height(), image.width(), 1);
		for (size_t j = 0; j < image.width() * image.height(); ++j)
		{
			inverse_two_dim_fft(j % image.width(), j / image.height(), 0, i) = channels[i][j];
		}
	}
	cimg_library::CImgDisplay original_display(image, "original"), two_dim_fft_display(two_dim_fft_res, "2DFFT"), two_dim_fft_display_cut_display(two_dim_fft_res_cut, "Low pass"), inverse_two_dim_fft_display(inverse_two_dim_fft, "Reversed");
	getchar();
	return 0;
}