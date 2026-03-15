const esbuild = require('esbuild');
const JavaScriptObfuscator = require('javascript-obfuscator');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const srcDir = path.join(__dirname, 'src');
const publicDir = path.join(__dirname, 'public');
const distDir = path.join(__dirname, 'dist');
const chunksDir = path.join(distDir, 'static', 'chunks');

// 1. Clean and setup directories
fs.rmSync(distDir, { recursive: true, force: true });
fs.mkdirSync(chunksDir, { recursive: true });

async function build() {
    try {
        // 2. Bundle & Minify with esbuild
        const result = await esbuild.build({
            entryPoints: [path.join(srcDir, 'main.js')],
            bundle: true,
            minify: true,
            write: false,
            format: 'iife' // Immediately Invoked Function Expression for vanilla JS
        });

        const minifiedCode = result.outputFiles[0].text;

        // 3. Obfuscate the bundled code
        const obfuscatedResult = JavaScriptObfuscator.obfuscate(minifiedCode, {
            compact: true,
            controlFlowFlattening: true,
            controlFlowFlatteningThreshold: 0.75,
            numbersToExpressions: true,
            simplify: true,
            stringArrayThreshold: 0.75,
            deadCodeInjection: false
        });

        const finalCode = obfuscatedResult.getObfuscatedCode();

        // 4. Generate Hash and Write File
        const hash = crypto.createHash('md5').update(finalCode).digest('hex').substring(0, 8);
        const fileName = `app.${hash}.js`;
        fs.writeFileSync(path.join(chunksDir, fileName), finalCode);

        // 5. Process index.html
        let html = fs.readFileSync(path.join(publicDir, 'index.html'), 'utf-8');
        
        // Strip out any existing development script tags dynamically
        html = html.replace(/<script src="\.\.\/src\/scripts\/.*?\.js"><\/script>\n?/g, '');
        html = html.replace(/<script src="\.\.\/src\/script\.js"><\/script>\n?/g, '');
        html = html.replace(/<script src="\.\.\/src\/main\.js"><\/script>\n?/g, '');
        
        // Inject the production hashed script before the closing </body> tag
        html = html.replace('</body>', `    <script src="static/chunks/${fileName}"></script>\n</body>`);
        fs.writeFileSync(path.join(distDir, 'index.html'), html);

        // 6. Copy static assets
        if (fs.existsSync(path.join(publicDir, 'styles.css'))) {
            fs.copyFileSync(path.join(publicDir, 'styles.css'), path.join(distDir, 'styles.css'));
        }

        // FIX: Recursively copy the missing 'assets' directory to dist
        const assetsSrc = path.join(__dirname, 'assets');
        const assetsDest = path.join(distDir, 'assets');
        
        function copyDirectorySync(src, dest) {
            if (!fs.existsSync(src)) return;
            fs.mkdirSync(dest, { recursive: true });
            const entries = fs.readdirSync(src, { withFileTypes: true });
            for (let entry of entries) {
                const srcPath = path.join(src, entry.name);
                const destPath = path.join(dest, entry.name);
                if (entry.isDirectory()) {
                    copyDirectorySync(srcPath, destPath);
                } else {
                    fs.copyFileSync(srcPath, destPath);
                }
            }
        }
        
        copyDirectorySync(assetsSrc, assetsDest);

        console.log(`✅ Build successful. Obfuscated asset: static/chunks/${fileName}`);
    } catch (err) {
        console.error('❌ Build failed:', err);
        process.exit(1);
    }
}

build();