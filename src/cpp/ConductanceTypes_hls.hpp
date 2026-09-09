#pragma once

// HLS-side conductance types.
// Use this header for Bambu or HDL-oriented builds. Replace scalar_t here with
// a fixed-point type when you are ready to synthesize the conductance models.

#include <algorithm>
#include <cmath>
#include <cstdint>

using scalar_t = double;

struct StochKVTransitions {
	int n0_to_n1;
	int n1_to_n0;
};

inline std::uint32_t ConductanceTypes_hls_next_u32(std::uint32_t& state) {
	std::uint32_t x = state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	state = x;
	return x;
}

inline scalar_t ConductanceTypes_hls_uniform01(std::uint32_t& state) {
	return static_cast<scalar_t>(ConductanceTypes_hls_next_u32(state)) / 4294967296.0;
}

inline int ConductanceTypes_hls_sample_binomial(int trials, scalar_t probability, std::uint32_t& rng_state) {
	int successes = 0;
	for (int i = 0; i < trials; ++i) {
		if (ConductanceTypes_hls_uniform01(rng_state) < probability) {
			++successes;
		}
	}
	return successes;
}

inline StochKVTransitions StochKV_sample_transitions(scalar_t n, scalar_t V, scalar_t celsius, scalar_t area, scalar_t dt, std::uint32_t& rng_state) {
	constexpr scalar_t THA = -40.0;
	constexpr scalar_t QA = 9.0;
	constexpr scalar_t RA = 0.02;
	constexpr scalar_t RB = 0.002;
	constexpr scalar_t Q10 = 2.3;
	constexpr scalar_t TEMP = 23.0;
	constexpr scalar_t e = 2.718281828459045235360287471352;
	constexpr scalar_t gK_bar = 0.75;
	constexpr scalar_t gamma = 30.0;

	const scalar_t tadj = std::pow(Q10, ((celsius - TEMP) / 10.0));

	scalar_t a = 0.0;
	scalar_t b = 0.0;
	if (std::fabs(V - THA) > 1e-6) {
		a = RA * (V - THA) / (1.0 - std::pow(e, ((THA - V) / QA)));
		b = -RB * (V - THA) / (1.0 - std::pow(e, ((THA - V) / QA)));
	} else {
		a = RA * QA;
		b = RB * QA;
	}
	a *= tadj;
	b *= tadj;

	const scalar_t eta = gK_bar / gamma;
	const int N = static_cast<int>(std::floor(eta * area + 0.5));
	const int N1 = static_cast<int>(std::floor(n * N + 0.5));
	const int N0 = N - N1;

	const scalar_t P_a = std::max<scalar_t>(0.0, std::min<scalar_t>(1.0, a * dt));
	const scalar_t P_b = std::max<scalar_t>(0.0, std::min<scalar_t>(1.0, b * dt));

	return {ConductanceTypes_hls_sample_binomial(N0, P_a, rng_state), ConductanceTypes_hls_sample_binomial(N1, P_b, rng_state)};
}