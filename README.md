# Notion Curriculum Sync

Automatically sync the "2-Week Python & C++ Proficiency for Statisticians and Data Scientists" curriculum from Notion to this GitHub repository as Markdown.

## Features

- **Automated sync**: GitHub Actions runs weekly (configurable)
- **Manual trigger**: Sync on-demand from GitHub Actions UI
- **Full Markdown export**: Code blocks, tables, headings preserved
- **Change detection**: Only commits when content changes
- **Metadata tracking**: Front matter includes sync timestamps and Notion URL

## Quick Start

### 1. Create a Notion Integration

1. Go to [Notion Integrations](https://www.notion.so/my-integrations)
2. Click **"+ New integration"**
3. Name it (e.g., "GitHub Sync")
4. Select your workspace
5. Click **Submit**
6. Copy the **Internal Integration Token** (starts with `secret_`)

### 2. Share Your Notion Page with the Integration

1. Open your curriculum page in Notion
2. Click **Share** (top right)
3. Click **"Invite"**
4. Select your integration from the dropdown
5. Click **Invite**

### 3. Add Secrets to GitHub

1. Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions**
2. Add these secrets:

| Secret Name | Value |
|-------------|-------|
| `NOTION_TOKEN` | Your integration token (`secret_xxx...`) |
| `NOTION_PAGE_ID` | `2db342cf7cc8815a97e5d434dbabf57c` |

### 4. Run the Sync

**Option A: Manual trigger**
1. Go to **Actions** tab in GitHub
2. Select **"Sync Notion Curriculum"**
3. Click **"Run workflow"**

**Option B: Wait for scheduled run**
- Runs automatically every Sunday at 6 AM UTC

## Local Development

### Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd notion-curriculum-sync

# Install dependencies
npm install

# Create environment file
cp .env.example .env
# Edit .env with your NOTION_TOKEN
```

### Run Export Locally

```bash
# Export to ./curriculum/
npm run export

# Preview without writing (dry run)
npm run export:dry-run
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NOTION_TOKEN` | Yes | - | Notion integration token |
| `PAGE_ID` | No | Curriculum page | Notion page ID to export |
| `OUTPUT_DIR` | No | `./curriculum` | Output directory |
| `DRY_RUN` | No | `false` | Preview mode |

## Output

The exported curriculum is saved to:

```
curriculum/
└── python-cpp-proficiency-curriculum.md
```

The Markdown file includes YAML front matter:

```yaml
---
title: "2-Week Python & C++ Proficiency for Statisticians and Data Scientists"
source: Notion
notion_url: https://www.notion.so/...
last_synced: 2025-01-02T12:00:00.000Z
last_edited_in_notion: 2025-01-02T10:30:00.000Z
---
```

## Customization

### Change Sync Schedule

Edit `.github/workflows/sync-notion.yml`:

```yaml
on:
  schedule:
    # Every Sunday at 6 AM UTC (default)
    - cron: '0 6 * * 0'
    
    # Daily at midnight UTC
    # - cron: '0 0 * * *'
    
    # Every 6 hours
    # - cron: '0 */6 * * *'
```

### Export Multiple Pages

Modify `scripts/export-notion.js` to export multiple pages:

```javascript
const PAGES = [
  { id: 'xxx', output: 'curriculum-python-cpp.md' },
  { id: 'yyy', output: 'another-doc.md' },
];
```

## Troubleshooting

### "Could not find block"
- Ensure the integration has access to the page
- Check if the page ID is correct (from the URL)

### "Unauthorized"
- Verify `NOTION_TOKEN` is correct
- Ensure the integration is still active

### Export is incomplete
- Large pages may take longer; check GitHub Actions logs
- Notion API has rate limits; the script handles these automatically

### Tables look wrong
- Notion's table format differs from GitHub Markdown
- Complex tables may need manual cleanup

## File Structure

```
.
├── .github/
│   └── workflows/
│       └── sync-notion.yml    # GitHub Actions workflow
├── curriculum/
│   └── python-cpp-proficiency-curriculum.md  # Exported content
├── scripts/
│   └── export-notion.js       # Export script
├── .env.example               # Environment template
├── .gitignore
├── package.json
└── README.md
```

## License

MIT
