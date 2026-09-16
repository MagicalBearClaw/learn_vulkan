// stb's headers carry their own implementations behind a macro, which has to be
// compiled in exactly one translation unit. This file is that place for the *writing*
// half, and like vma_impl.cpp it deliberately contains nothing else.
//
// stb_image -- the reading half -- is deliberately not here. Assimp bundles its own copy
// of stb_image and its static library exports the same stbi_* names, so a second global
// definition in this library makes every model-loading chapter fail to link with
// "multiple definition of stbi_load". The two translation units that actually read an
// image file -- vkcommon's image.cpp and chapter 1.11 -- compile their own copy with
// STB_IMAGE_STATIC, which no other library can see or collide with.
//
// Nothing exports stbi_write_*, so the writing half stays here, shared as it always was.

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
