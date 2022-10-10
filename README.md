# TDA
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7181486.svg)](https://doi.org/10.5281/zenodo.7181486)

This tool called TDA (Traffic Dynamics Analyser) extracts data from [the PeMS database](dot.ca.gov/programs/traffic-operations/mpr/pems-source), processes this data and generates graphs that visualize the macroscopic traffic dynamics. It also fits the data to established models.

The creator assumes no responsibility or liability in connection with the information or opinions contained in or expressed by this tool, its use or output.

## License
TDA is licensed under the GNU General Public License v3.0.

## Disclaimer
Important Notice and Disclaimers before downloading the TDA:
By accessing, downloading and using the TDA, you expressly agree to the following notices and disclaimers, as well as the End User License Agreement found here.

The tool extracts data from [the PeMS database](dot.ca.gov/programs/traffic-operations/mpr/pems-source), processes this data and generates graphs that visualize the macroscopic traffic dynamics. It also fits the data to established models. 
This tool and any related suggestions around evacuation are not designed to replace or substitute an AHJ’s decision about evacuation during a wildfire.
Use of this tool is at the user’s own risk; it is provided AS IS and AS AVAILABLE without guarantee or warranty of any kind, express or implied (including the warranties of merchantability and fitness for a particular purpose) and without representation or warranty regarding its accuracy, completeness, usefulness, timeliness, reliability or appropriateness. 

## Installation guide
1. Install python
2. Make sure the required packages are installed:
   [os](https://docs.python.org/3/library/os.html), [datetime](https://docs.python.org/3/library/datetime.html), [requests](https://pypi.org/project/requests/), [numpy](https://numpy.org/), [scipy](https://scipy.org/), [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/), [threading](https://docs.python.org/3/library/threading.html), [kivy](https://kivy.org/doc/stable/gettingstarted/installation.html)
3. Run `main.py`
4. Read more about how to use the tool [in the user guide](https://github.com/Division-of-Fire-Safety-Engineering-LU/TDA/blob/main/TDA%20User%20Guide.pdf), or [watch this tutorial on YouTube](https://youtu.be/QvHip4qKFhM)
