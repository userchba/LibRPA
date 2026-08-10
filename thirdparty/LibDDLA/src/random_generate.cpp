#include <ddla/random_generate.h>

#include <chrono>
#include <random>
#include <type_traits>

namespace ddla {

template <DdlaBackend Backend, typename T>
void random_generate(T* data, const int64_t& lengthOfData)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU random_generate is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU random_generate is not available in this LibDDLA build");

    if (lengthOfData <= 0) return;
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();

    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        std::mt19937_64 rng(static_cast<unsigned long long>(seed));
        if constexpr (std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (int64_t i = 0; i < lengthOfData; ++i) data[i] = dist(rng);
        } else if constexpr (std::is_same_v<T, double>) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (int64_t i = 0; i < lengthOfData; ++i) data[i] = dist(rng);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float* p = reinterpret_cast<float*>(data);
            for (int64_t i = 0; i < 2 * lengthOfData; ++i) p[i] = dist(rng);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double* p = reinterpret_cast<double*>(data);
            for (int64_t i = 0; i < 2 * lengthOfData; ++i) p[i] = dist(rng);
        } else {
            static_assert(sizeof(T) == 0,
                          "random_generate supports only float, double, std::complex<float>, std::complex<double>");
        }
#endif
    } else {
#if DDLA_HAS_GPU
        derandGenerator_t gen;
        DERAND_CHECK(derandCreateGenerator(&gen, DERAND_RNG_PSEUDO_DEFAULT));
        DERAND_CHECK(derandSetPseudoRandomGeneratorSeed(gen, static_cast<unsigned long long>(seed)));
        if constexpr (std::is_same_v<T, float>) {
            DERAND_CHECK(derandGenerateUniform(gen, data, lengthOfData));
        } else if constexpr (std::is_same_v<T, double>) {
            DERAND_CHECK(derandGenerateUniformDouble(gen, data, lengthOfData));
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            DERAND_CHECK(derandGenerateUniform(gen, reinterpret_cast<float*>(data), lengthOfData * 2));
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            DERAND_CHECK(derandGenerateUniformDouble(gen, reinterpret_cast<double*>(data), lengthOfData * 2));
        } else {
            static_assert(sizeof(T) == 0,
                          "random_generate supports only float, double, std::complex<float>, std::complex<double>");
        }
        DERAND_CHECK(derandDestroyGenerator(gen));
#endif
    }
}

#define INSTANTIATE_RANDOM_GENERATE(BACKEND, TYPE)                           \
    template void random_generate<BACKEND, TYPE>(TYPE*, const int64_t&)

#if DDLA_HAS_CPU
INSTANTIATE_RANDOM_GENERATE(DdlaBackend::CPU, float);
INSTANTIATE_RANDOM_GENERATE(DdlaBackend::CPU, double);
INSTANTIATE_RANDOM_GENERATE(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_RANDOM_GENERATE(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_RANDOM_GENERATE(DdlaBackend::GPU, float);
INSTANTIATE_RANDOM_GENERATE(DdlaBackend::GPU, double);
INSTANTIATE_RANDOM_GENERATE(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_RANDOM_GENERATE(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_RANDOM_GENERATE

} // namespace ddla
