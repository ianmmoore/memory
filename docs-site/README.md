# Memory API Documentation Site

Customer-facing documentation built with Docusaurus.

## Development

```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build

# Serve production build locally
npm run serve
```

## Deployment

### Vercel (Recommended)

1. Connect your GitHub repository to Vercel
2. Set build command: `npm run build`
3. Set output directory: `build`
4. Deploy!

### Netlify

1. Connect repository
2. Build command: `npm run build`
3. Publish directory: `build`

### Docker

```bash
# Build
docker build -t memory-api-docs .

# Run
docker run -p 3000:3000 memory-api-docs
```

### GitHub Pages

```bash
USE_SSH=true npm run deploy
```

## Structure

```
docs/
├── index.md                 # Home page
├── getting-started/         # Onboarding guides
├── api-reference/           # Endpoint docs
├── sdks/                    # SDK guides
├── guides/                  # How-to guides
├── pricing.md               # Pricing page
└── changelog.md             # Version history
```

## Customization

- **Theme**: Edit `docusaurus.config.js`
- **Styles**: Edit `src/css/custom.css`
- **Sidebar**: Edit `sidebars.js`

## Adding Content

1. Create markdown file in `docs/`
2. Add frontmatter with `sidebar_position`
3. Update `sidebars.js` if needed
4. Commit and deploy

## Search

To enable search, configure Algolia in `docusaurus.config.js`:

```js
algolia: {
  appId: 'YOUR_APP_ID',
  apiKey: 'YOUR_API_KEY',
  indexName: 'memory-api',
}
```
