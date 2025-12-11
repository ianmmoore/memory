/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docsSidebar: [
    {
      type: 'doc',
      id: 'index',
      label: 'Introduction',
    },
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/quickstart',
        'getting-started/authentication',
        'getting-started/first-request',
        'getting-started/concepts',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      collapsed: false,
      items: [
        'api-reference/overview',
        'api-reference/memories',
        'api-reference/query',
        'api-reference/account',
        'api-reference/usage',
        'api-reference/errors',
      ],
    },
    {
      type: 'category',
      label: 'SDKs',
      items: [
        'sdks/python',
        'sdks/javascript',
        'sdks/rest',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      items: [
        'guides/memory-extraction',
        'guides/retrieval-tuning',
        'guides/best-practices',
      ],
    },
    {
      type: 'doc',
      id: 'pricing',
      label: 'Pricing',
    },
    {
      type: 'doc',
      id: 'changelog',
      label: 'Changelog',
    },
  ],
};

module.exports = sidebars;
