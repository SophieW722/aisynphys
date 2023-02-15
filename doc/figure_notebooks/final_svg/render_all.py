import os, sys, glob, subprocess

if len(sys.argv) == 1:
    print("Usage:  python render_all.py <png|pdf>")
    sys.exit(1)
out_type = sys.argv[1]
assert out_type in ['png', 'pdf']

if not os.path.exists('export'):
    os.mkdir('export')

for svg_file in sorted(glob.glob('*.svg')):
    out_file = os.path.join('export', os.path.splitext(svg_file)[0] + '.' + out_type)
    if os.path.exists(out_file) and os.stat(out_file).st_mtime > os.stat(svg_file).st_mtime:
        continue
    print(f"{svg_file} ⇒ {out_file}")
    subprocess.check_output([
        'inkscape', 
        '--export-filename', out_file,
        '--export-type', out_type,
        '--export-area-page', 
        '--export-dpi=1200', 
        '--export-background=white',
        svg_file, 
    ])
