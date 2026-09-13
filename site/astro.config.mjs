// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// The sidebar is the curriculum. Parts are listed in reading order and each one is
// an `autogenerate` directory, so adding a chapter page is all it takes to add it to
// the navigation -- `sidebar.order` in the page frontmatter places it within its part.
export default defineConfig({
  site: "https://example.com",
  integrations: [
    starlight({
      title: "LearnVulkan",
      description:
        "A complete, beginner-oriented course in modern Vulkan: from opening a window to physically based rendering.",
      customCss: ["./src/styles/custom.css"],
      credits: false,
      tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
      lastUpdated: true,
      pagination: true,
      sidebar: [
        {
          label: "Introduction",
          collapsed: false,
          items: [{ autogenerate: { directory: "introduction" } }],
        },
        {
          label: "Getting Started",
          collapsed: false,
          items: [{ autogenerate: { directory: "getting-started" } }],
        },
        {
          label: "Lighting",
          collapsed: true,
          items: [{ autogenerate: { directory: "lighting" } }],
        },
        {
          label: "Model Loading",
          collapsed: true,
          items: [{ autogenerate: { directory: "model-loading" } }],
        },
        {
          label: "Advanced Vulkan",
          collapsed: true,
          items: [{ autogenerate: { directory: "advanced-vulkan" } }],
        },
        {
          label: "Advanced Lighting",
          collapsed: true,
          items: [{ autogenerate: { directory: "advanced-lighting" } }],
        },
        {
          label: "PBR",
          collapsed: true,
          items: [{ autogenerate: { directory: "pbr" } }],
        },
        {
          label: "In Practice",
          collapsed: true,
          items: [{ autogenerate: { directory: "in-practice" } }],
        },
        {
          label: "Extra Topics",
          collapsed: true,
          items: [{ autogenerate: { directory: "extra" } }],
        },
        {
          label: "Vulkan Only",
          collapsed: true,
          items: [{ autogenerate: { directory: "vulkan-only" } }],
        },
        {
          label: "Appendices",
          collapsed: true,
          items: [{ autogenerate: { directory: "appendices" } }],
        },
      ],
    }),
  ],
});
