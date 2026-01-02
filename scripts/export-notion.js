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

  return processed;
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

  // Write output
  writeOutput(finalMarkdown, metadata);

  console.log('\n✓ Export complete!');
}

// Run
main().catch((error) => {
  console.error('\nExport failed:', error);
  process.exit(1);
});
