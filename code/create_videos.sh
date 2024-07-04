set -e

exp_id=$1

cd logs/$exp_id/test/
mkdir -p animations
ffmpeg -framerate 5 -pattern_type glob -i './visuals/rgb/*.png' -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" animations/rgb.mp4 
ffmpeg -framerate 5 -pattern_type glob -i './visuals/normal/*.png' -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" animations/normal.mp4 
ffmpeg -framerate 5 -pattern_type glob -i './visuals/imap/*.png' -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" animations/imap.mp4 
ffmpeg -framerate 5 -pattern_type glob -i './visuals/bg_rgb/*.png' -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" animations/bg_rgb.mp4 
echo "Videos created in logs/$exp_id/test/animations/"