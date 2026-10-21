// Keep DSH's native file tools and mutation guards, adding RAX's read boundary.
import { readFileSync } from 'node:fs';
import { isAbsolute, relative } from 'node:path';
import { pathToFileURL } from 'node:url';

const base = process.env.RAX_DSH_INSTALL;
const { SandboxedFileSystem } = await import(pathToFileURL(`${base}/node_modules/@deepseek-ai/dsh-fs-sandbox/lib/index.js`).href);
const { FsError } = await import(pathToFileURL(`${base}/node_modules/@deepseek-ai/dsh-fs/lib/index.js`).href);
const within = (path, root) => {
  const value = relative(root, path);
  return value === '' || (!isAbsolute(value) && value !== '..' && !value.startsWith('../'));
};

export default class RaxFileSystem extends SandboxedFileSystem {
  constructor(ctx, config) {
    super(ctx, config);
    this.roots = JSON.parse(readFileSync(process.env.RAX_DSH_BOUNDARY, 'utf8')).file_read_roots;
  }

  async resolve(path) {
    const target = await super.resolve(path);
    if (!this.roots.some(root => within(target.targetKey, root))) {
      throw new FsError('This path is outside the experiment file boundary', 'FS_SANDBOX_DENIED');
    }
    return target;
  }
}
