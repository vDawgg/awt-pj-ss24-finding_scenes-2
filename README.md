# awt-pj-ss24-finding_scenes-2

## Setup


To install the project's dependencies, run the following command in your terminal:

```bash
pip install -r requirements.txt
```



As with most python projects we recommend setting up a [virtual environment](https://docs.python.org/3/library/venv.html).

## Dependencies

To install the non python dependencies of the project run the following, depending on your systen:

### Arch

`sudo pacman -S ffmpeg`

### Ubuntu

`sudo apt-get install ffmpeg`



```bash
apt install imagemagick
```

Additionally, ensure ImageMagick permissions are properly configured by running:

```bash
cat /etc/ImageMagick-6/policy.xml | sed 's/none/read,write/g'> /etc/ImageMagick-6/policy.xml
```
=======

## Running

To run the current implementation of the pipeline open the colab_pipeline in colab or run it locally.
