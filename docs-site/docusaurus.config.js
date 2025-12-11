// @ts-check
const {themes} = require('prism-react-renderer');
const lightCodeTheme = themes.github;
const darkCodeTheme = themes.dracula;

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Memory API',
  tagline: 'Intelligent memory storage and retrieval for AI applications',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://docs.memory-api.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  baseUrl: '/',

  // GitHub pages deployment config (if using)
  organizationName: 'your-org',
  projectName: 'memory-api-docs',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          routeBasePath: '/', // Serve docs at root
        },
        blog: false, // Disable blog
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Social card image
      image: 'img/social-card.png',
      navbar: {
        title: 'Memory API',
        logo: {
          alt: 'Memory API Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docsSidebar',
            position: 'left',
            label: 'Documentation',
          },
          {
            href: 'https://api.memory-api.com/docs',
            label: 'API Explorer',
            position: 'left',
          },
          {
            href: 'https://dashboard.memory-api.com',
            label: 'Dashboard',
            position: 'right',
          },
          {
            href: 'https://github.com/your-org/memory-api',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Documentation',
            items: [
              {
                label: 'Getting Started',
                to: '/getting-started',
              },
              {
                label: 'API Reference',
                to: '/api-reference',
              },
              {
                label: 'SDKs',
                to: '/sdks',
              },
            ],
          },
          {
            title: 'Resources',
            items: [
              {
                label: 'Pricing',
                to: '/pricing',
              },
              {
                label: 'Status',
                href: 'https://status.memory-api.com',
              },
              {
                label: 'Changelog',
                to: '/changelog',
              },
            ],
          },
          {
            title: 'Company',
            items: [
              {
                label: 'About',
                href: 'https://memory-api.com/about',
              },
              {
                label: 'Blog',
                href: 'https://memory-api.com/blog',
              },
              {
                label: 'Contact',
                href: 'mailto:support@memory-api.com',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Memory API. All rights reserved.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
        additionalLanguages: ['bash', 'json', 'python', 'javascript', 'yaml'],
      },
      // Algolia search (configure with your own credentials)
      // algolia: {
      //   appId: 'YOUR_APP_ID',
      //   apiKey: 'YOUR_API_KEY',
      //   indexName: 'memory-api',
      // },
      colorMode: {
        defaultMode: 'light',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
    }),
};

module.exports = config;
