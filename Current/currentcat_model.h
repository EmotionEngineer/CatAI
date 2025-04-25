#ifndef CURRENTCAT_MODEL_H_
#define CURRENTCAT_MODEL_H_

#include "currentcat.h"

#include "currentcat_tensor.h"

LIB_HIDDEN void
model_infer (struct tensors *ts,
             ModelInput const *images,
             ModelOut *result);

#endif // CURRENTCAT_MODEL_H_
