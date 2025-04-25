#include <stdio.h>
#include <string.h>

#include "currentcat_model.h"

__attribute__((always_inline))
static inline struct tensors *
tensors_create_helper (char const *const caller)
{
	int             e = 0;
	struct tensors *t = tensors_create(&e);

	if (!t) {
		if (e)
			fprintf(stderr, "%s: tensors_create(): %s\n", caller, strerror(e));
		else
			fprintf(stderr, "%s: tensors_create() failed\n", caller);
	}

	return t;
}

LIB_EXPORT void
currentcat_Inference (ModelInput const *images,
                       ModelOut  *result)
{
	struct tensors *ts = tensors_create_helper(__func__);

	if (ts) {
		model_infer(ts, images, result);
		tensors_destroy(&ts);
	}
}
