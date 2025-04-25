#ifndef CATNEXT_MODEL_H_
#define CATNEXT_MODEL_H_

#include "catnext.h"

#include "catnext_tensor.h"

LIB_HIDDEN void
model_infer (struct tensors *ts,
             ModelInput const *images,
             ModelAddInput const *info,
             ModelOut *result);

#endif // CATNEXT_MODEL_H_
