# awt-pj-ss24-finding_scenes-2

## Setup

To install the project's dependencies, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

Python projects typically benefit from using virtual environments to manage dependencies. If you're unfamiliar with virtual environments, check out the [official documentation](https://docs.python.org/3/library/venv.html) for guidance.

### ImageMagick Installation

#### For Linux Users:

Linux users can install ImageMagick using the package manager. Simply execute the following command in your terminal:

```bash
apt install imagemagick
```

Additionally, ensure ImageMagick permissions are properly configured by running:

```bash
cat /etc/ImageMagick-6/policy.xml | sed 's/none/read,write/g'> /etc/ImageMagick-6/policy.xml
```

#### For Windows Users:

Windows users should follow these steps:

1. **Download**: Get the ImageMagick installer from the official website: [ImageMagick Downloads](https://imagemagick.org/script/download.php)

2. **Installation**: Run the installer and follow the on-screen instructions.

3. **Add to PATH**: After installation, add ImageMagick to your system's PATH environment variable. You can typically do this during installation or manually afterwards. Refer to online tutorials or documentation specific to your Windows version for guidance.

Once these steps are completed, you'll be ready to effectively use the project. If you encounter any issues during installation, feel free to ask for further assistance!