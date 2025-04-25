#ifndef CATNEXT_H_
#define CATNEXT_H_

#if defined(_MSC_VER) || defined(_WIN32)
# define CATNEXT_API_EXPORT  __declspec(dllexport)
#else
# define CATNEXT_API_EXPORT  __attribute__((visibility("default")))
#endif

typedef float ModelInput[1][384][384][39];
typedef float ModelAddInput[1][300];
typedef float ModelOut[1][66];

// Model inference
CATNEXT_API_EXPORT void
catnext_Inference (ModelInput const *images,
                   ModelAddInput const *info,
                   ModelOut  *result);

#undef CATNEXT_API_EXPORT

#endif // CATNEXT_H_
