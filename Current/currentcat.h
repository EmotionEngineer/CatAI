#ifndef CURRENTCAT_H_
#define CURRENTCAT_H_

#if defined(_MSC_VER) || defined(_WIN32)
# define CURRENTCAT_API_EXPORT  __declspec(dllexport)
#else
# define CURRENTCAT_API_EXPORT  __attribute__((visibility("default")))
#endif

typedef float ModelInput[1][384][384][39];
typedef float ModelOut[1][66];

// Model inference
CURRENTCAT_API_EXPORT void
currentcat_Inference (ModelInput const *images,
                       ModelOut  *result);

#undef CURRENTCAT_API_EXPORT

#endif // CURRENTCAT_H_
