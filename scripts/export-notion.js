/**
 * Export Notion page to Markdown
 *
 * Specifically designed for the "2-Week Python & C++ Proficiency" curriculum.
 * Handles large documents, code blocks, tables, and nested content.
 *
 * Usage:
 *   NOTION_TOKEN=secret_xxx PAGE_ID=xxx node scripts/export-notion.js
 *
 * Environment variables:
 *   NOTION_TOKEN - Notion integration token (required)
 *   PAGE_ID      - Notion page ID to export (default: curriculum page)
 *   OUTPUT_DIR   - Output directory (default: ./curriculum)
 *   DRY_RUN      - If "true", print to stdout instead of writing file
 *
 * Post-processing features:
 *   - Fixes malformed Notion URL artifacts (%7B%7B encoding)
 *   - Converts Notion internal links to markdown anchors
 *   - Detects and fixes incorrect language tags (e.g., javascript for ASCII diagrams)
 *   - Removes duplicate consecutive Solution Sketch blocks
 *   - Validates output for truncated code blocks and other issues
 */

import { Client } from '@notionhq/client';
import { NotionToMarkdown } from 'notion-to-md';
import fs from 'fs';
import path from 'path';

// Configuration
const CONFIG = {
  notionToken: process.env.NOTION_TOKEN,
  pageId: process.env.PAGE_ID || '2db342cf7cc8815a97e5d434dbabf57c',
  outputDir: process.env.OUTPUT_DIR || './curriculum',
  outputFile: 'python-cpp-proficiency-curriculum.md',
  dryRun: process.env.DRY_RUN === 'true',
};

// Validate configuration
function validateConfig() {
  if (!CONFIG.notionToken) {
    console.error('Error: NOTION_TOKEN environment variable is required');
    console.error('');
    console.error('To get a token:');
    console.error('  1. Go to https://www.notion.so/my-integrations');
    console.error('  2. Create a new integration');
    console.error('  3. Copy the "Internal Integration Token"');
    console.error('  4. Share your Notion page with the integration');
    process.exit(1);
  }
}

// Initialize clients
function initClients() {
  const notion = new Client({ auth: CONFIG.notionToken });
  const n2m = new NotionToMarkdown({ 
    notionClient: notion,
    config: {
      separateChildPage: false,  // Include child pages inline
    }
  });

  // Custom transformer for code blocks to preserve language
  n2m.setCustomTransformer('code', async (block) => {
    const { code } = block;
    const language = code.language || 'text';
    const text = code.rich_text.map(t => t.plain_text).join('');
    return `\`\`\`${language}\n${text}\n\`\`\``;
  });

  // Custom transformer for tables to ensure proper formatting
  n2m.setCustomTransformer('table', async (block) => {
    // Let default handler process, but ensure blank lines around tables
    return false; // Use default
  });

  return { notion, n2m };
}

// Fetch page metadata
async function getPageMetadata(notion, pageId) {
  try {
    const page = await notion.pages.retrieve({ page_id: pageId });
    
    // Extract title
    let title = 'Untitled';
    const titleProp = page.properties?.title || page.properties?.Name;
    if (titleProp?.title?.[0]?.plain_text) {
      title = titleProp.title[0].plain_text;
    }

    return {
      title,
      lastEdited: page.last_edited_time,
      url: page.url,
    };
  } catch (error) {
    console.error('Error fetching page metadata:', error.message);
    throw error;
  }
}

// Export page to markdown
async function exportPage(n2m, pageId) {
  console.log(`Exporting page ${pageId}...`);
  
  try {
    const mdBlocks = await n2m.pageToMarkdown(pageId);
    const mdString = n2m.toMarkdownString(mdBlocks);
    return mdString.parent;
  } catch (error) {
    console.error('Error exporting page:', error.message);
    throw error;
  }
}

// Post-process markdown for better formatting
function postProcessMarkdown(markdown, metadata) {
  let processed = markdown;

  // Add front matter with metadata
  const frontMatter = `---
title: "${metadata.title}"
source: Notion
notion_url: ${metadata.url}
last_synced: ${new Date().toISOString()}
last_edited_in_notion: ${metadata.lastEdited}
---

`;

  processed = frontMatter + processed;

  // Fix common issues

  // 1. Ensure blank lines before code blocks
  processed = processed.replace(/([^\n])\n```/g, '$1\n\n```');

  // 2. Ensure blank lines after code blocks
  processed = processed.replace(/```\n([^\n])/g, '```\n\n$1');

  // 3. Fix heading spacing
  processed = processed.replace(/([^\n])\n(#{1,6} )/g, '$1\n\n$2');

  // 4. Remove excessive blank lines (more than 2)
  processed = processed.replace(/\n{4,}/g, '\n\n\n');

  // 5. Fix table formatting - ensure blank line before tables
  processed = processed.replace(/([^\n])\n\|/g, '$1\n\n|');

  // 6. Convert Notion-style callouts to blockquotes with emoji
  processed = processed.replace(/> (⚠️|💡|📝|✅|❌|🔥|📌)/g, '> **$1**');

  // 7. Fix malformed Notion URL artifacts from inline code references
  // Pattern: [`](%7B%7Bhttp://filename.py%7D%7D)[filename.py](http://filename.py/)[`](%7B%7Bhttp://filename.py%7D%7D)
  // Should become: `filename.py`
  processed = processed.replace(
    /\[`\]\(%7B%7Bhttp:\/\/([^%]+)%7D%7D\)\[[^\]]+\]\([^)]+\)\[`\]\(%7B%7Bhttp:\/\/[^%]+%7D%7D\)/g,
    '`$1`'
  );

  // Also handle simpler malformed patterns
  processed = processed.replace(
    /\[`\]\(%7B%7B[^)]+%7D%7D\)/g,
    '`'
  );

  // 8. Convert Notion internal page links to markdown anchors
  // Pattern: [Text](/pageId#blockId) or [Text](https://www.notion.so/...)
  processed = processed.replace(
    /\[([^\]]+)\]\(\/[a-f0-9]{32}#[a-f0-9]{32}\)/g,
    (match, text) => {
      // Convert text to anchor format: "Proficiency Standards" -> "#proficiency-standards"
      const anchor = text.toLowerCase().replace(/[^a-z0-9\s-]/g, '').replace(/\s+/g, '-');
      return `[${text}](#${anchor})`;
    }
  );

  // Also handle full Notion URLs
  processed = processed.replace(
    /\[([^\]]+)\]\(https:\/\/www\.notion\.so\/[^)]+\)/g,
    (match, text) => {
      const anchor = text.toLowerCase().replace(/[^a-z0-9\s-]/g, '').replace(/\s+/g, '-');
      return `[${text}](#${anchor})`;
    }
  );

  // 9. Fix javascript language tag for ASCII diagrams and file trees
  // Detect code blocks that contain directory structures or ASCII art
  processed = processed.replace(
    /```javascript\n((?:[\s\S]*?(?:├|└|│|─|project\/|src\/|include\/|tests\/|\.\.\.|\[\d+,\d+\])[\s\S]*?)?)```/g,
    (match, content) => {
      // Check if this looks like a directory tree or ASCII diagram (not actual JS)
      const looksLikeTree = /[├└│─]/.test(content) || /^\s*\w+\/\s*$/m.test(content);
      const looksLikeDiagram = /Buffer:|Shape:|Strides:|Array|Memory/.test(content);
      const hasNoJsSyntax = !/(?:function|const|let|var|=>|import|export|class)\s/.test(content);

      if ((looksLikeTree || looksLikeDiagram) && hasNoJsSyntax) {
        return '```text\n' + content + '```';
      }
      return match;
    }
  );

  // 10. Remove duplicate consecutive solution sketch blocks
  // Pattern: </details>\n\n<details>\n<summary>Solution Sketch</summary> appearing twice
  processed = removeDuplicateSolutionBlocks(processed);

  return processed;
}

// Remove duplicate solution sketch blocks that appear consecutively
function removeDuplicateSolutionBlocks(markdown) {
  const lines = markdown.split('\n');
  const result = [];
  let i = 0;

  while (i < lines.length) {
    result.push(lines[i]);

    // Check if we just closed a details block
    if (lines[i].trim() === '</details>') {
      // Look ahead to see if another Solution Sketch immediately follows
      let j = i + 1;

      // Skip blank lines
      while (j < lines.length && lines[j].trim() === '') {
        j++;
      }

      // Check if next non-blank content is another Solution Sketch details block
      if (j < lines.length &&
          lines[j].trim() === '<details>' &&
          j + 1 < lines.length &&
          lines[j + 1].includes('<summary>Solution Sketch</summary>')) {

        // Find the end of this duplicate block and skip it
        let depth = 1;
        let k = j + 2;
        while (k < lines.length && depth > 0) {
          if (lines[k].includes('<details>')) depth++;
          if (lines[k].includes('</details>')) depth--;
          k++;
        }

        console.log(`  Warning: Removed duplicate Solution Sketch block (lines ${j}-${k})`);

        // Skip to after the duplicate block
        i = k - 1; // -1 because the loop will increment
      }
    }

    i++;
  }

  return result.join('\n');
}

// Write output file
function writeOutput(content, metadata) {
  if (CONFIG.dryRun) {
    console.log('\n--- DRY RUN: Would write the following ---\n');
    console.log(content.substring(0, 2000));
    console.log(`\n... (${content.length} total characters)`);
    return;
  }

  // Ensure output directory exists
  if (!fs.existsSync(CONFIG.outputDir)) {
    fs.mkdirSync(CONFIG.outputDir, { recursive: true });
  }

  const outputPath = path.join(CONFIG.outputDir, CONFIG.outputFile);
  fs.writeFileSync(outputPath, content, 'utf-8');
  
  console.log(`\nWritten to: ${outputPath}`);
  console.log(`  Size: ${(content.length / 1024).toFixed(1)} KB`);
  console.log(`  Lines: ${content.split('\n').length}`);
}

// Generate export summary
function generateSummary(content) {
  const lines = content.split('\n');
  const codeBlocks = (content.match(/```/g) || []).length / 2;
  const headings = (content.match(/^#{1,6} /gm) || []).length;
  const tables = (content.match(/^\|/gm) || []).length;

  return {
    characters: content.length,
    lines: lines.length,
    codeBlocks: Math.floor(codeBlocks),
    headings,
    tables: Math.floor(tables / 3), // Rough estimate (header + separator + rows)
  };
}

// Validate exported content for common issues
function validateExport(content) {
  const issues = [];

  // Check for unbalanced code blocks
  const codeBlockMarkers = content.match(/```/g) || [];
  if (codeBlockMarkers.length % 2 !== 0) {
    issues.push('ERROR: Unbalanced code blocks (odd number of ``` markers)');
  }

  // Check for potentially truncated code blocks (common patterns that indicate incomplete code)
  const truncationPatterns = [
    { pattern: /return [a-zA-Z_][a-zA-Z0-9_.]*\s*\n```/g, name: 'truncated return statement' },
    { pattern: /def \w+\([^)]*\):\s*\n```/g, name: 'empty function body' },
    { pattern: /class \w+[^:]*:\s*\n```/g, name: 'empty class body' },
    { pattern: /{\s*\n```/g, name: 'empty block' },
    { pattern: /,\s*\n```/g, name: 'trailing comma before block end' },
    { pattern: /:\s*\n```/g, name: 'statement ending with colon' },
    { pattern: /for [^:]+\s*\n```/g, name: 'incomplete for loop' },
    { pattern: /if [^:]+\s*\n```/g, name: 'incomplete if statement' },
  ];

  for (const { pattern, name } of truncationPatterns) {
    const matches = content.match(pattern);
    if (matches) {
      for (const match of matches) {
        // Get context around the match
        const idx = content.indexOf(match);
        const lineNum = content.substring(0, idx).split('\n').length;
        issues.push(`WARNING: Possible ${name} at line ~${lineNum}`);
      }
    }
  }

  // Check for malformed links that weren't fixed
  if (content.includes('%7B%7B')) {
    const count = (content.match(/%7B%7B/g) || []).length;
    issues.push(`WARNING: ${count} malformed URL encoding artifacts remain`);
  }

  // Check for Notion page ID patterns that weren't converted
  const notionLinkPattern = /\]\(\/[a-f0-9]{32}/g;
  const notionLinks = content.match(notionLinkPattern);
  if (notionLinks) {
    issues.push(`WARNING: ${notionLinks.length} unconverted Notion internal links remain`);
  }

  return issues;
}

// Main export function
async function main() {
  console.log('Notion Curriculum Export');
  console.log('========================\n');

  validateConfig();

  const { notion, n2m } = initClients();

  // Get metadata
  console.log('Fetching page metadata...');
  const metadata = await getPageMetadata(notion, CONFIG.pageId);
  console.log(`  Title: ${metadata.title}`);
  console.log(`  Last edited: ${metadata.lastEdited}`);

  // Export to markdown
  console.log('\nConverting to Markdown...');
  const rawMarkdown = await exportPage(n2m, CONFIG.pageId);

  // Post-process
  console.log('Post-processing...');
  const finalMarkdown = postProcessMarkdown(rawMarkdown, metadata);

  // Generate summary
  const summary = generateSummary(finalMarkdown);
  console.log('\nExport summary:');
  console.log(`  Characters: ${summary.characters.toLocaleString()}`);
  console.log(`  Lines: ${summary.lines.toLocaleString()}`);
  console.log(`  Code blocks: ${summary.codeBlocks}`);
  console.log(`  Headings: ${summary.headings}`);

  // Validate export
  console.log('\nValidating export...');
  const issues = validateExport(finalMarkdown);
  if (issues.length > 0) {
    console.log('\n⚠️  Validation issues found:');
    for (const issue of issues) {
      console.log(`  ${issue}`);
    }
    console.log('');
  } else {
    console.log('  ✓ No issues detected');
  }

  // Write output
  writeOutput(finalMarkdown, metadata);

  // Final status
  const hasErrors = issues.some(i => i.startsWith('ERROR'));
  if (hasErrors) {
    console.log('\n⚠️  Export completed with errors - review output carefully');
    process.exit(1);
  } else if (issues.length > 0) {
    console.log('\n✓ Export complete (with warnings)');
  } else {
    console.log('\n✓ Export complete!');
  }
}

// Run
main().catch((error) => {
  console.error('\nExport failed:', error);
  process.exit(1);
});
