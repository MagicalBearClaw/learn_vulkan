// stb's headers carry their own implementations behind a macro, which has to be
// compiled in exactly one translation unit. This file is that place, and like
// vma_impl.cpp it deliberately contains nothing else.

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>
