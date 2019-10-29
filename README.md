# Density of representations yields age-related dissociations (DRYAD) model

Benjamin's (2010) DRYAD model
* Aaron Benjamin, University of Illinois at Urbana-Champaign

Citation:
* [Benjamin, A. S. (2010). Representational explanations of “process” dissociations in recognition: The DRYAD theory of aging and memory judgments. _Psychological Review, 117_(4), 1055.](http://psycnet.apa.org/record/2010-18184-001)


## DEPENDENCIES
While the goal is to provide an implementation of DRYAD that highly accessible and relatively free of dependencies, there are a few that are required to run this simulation.
* REQUIRED
   * Python 3.5+
   * NumPy (available through Anaconda suite or through Python; `pip install numpy` or `pip3 install numpy`
* OPTIONAL
   * Progress Bar (`conda install progressbar2`, `pip install progressbar2`, or `pip3 install progressbar2`)

## MAJOR UPDATES

### (092818)[TC]
* Code was continuing to under-estimate hit rate, but this error has been fixed. The code was only handling up to 6 context nodes - it can now handle much more than that.
* I am pretty confident that this is the final round of major code changes. Will be running Markov Chain analyses to compare to MATLAB output.

### (091418)[TC]
* Finished annotating the helper functions in `dryad_modules.py`. They will continue to be revised in order to better reflect what they are doing.
* Pushed changes to the `response` module in `dryad_modules.py`. It was under-estimating the hit rate due to a coding error in which it was evaluating hits to the second context - not the first.

### (090718)[TC]
* Modified the program to perform a grid search (i.e. loop through parameters). Also added a progress bar that shows how much time has elapsed as well as the approximate time left in the simulation. The progress bar can be suppressed by setting `see_progress_bar` to `0`.
* The progress bar requires installation of some outside packages. If you running the Anaconda/Spyder suite (which is recommended), pull up the Anaconda prompt and type in `conda install progressbar2`. If you are running Python, pull up a Python prompt window and type in either `pip install progressbar2` or `pip3 install progressbar1`, depending on which version of Python you are running.

![PRogress bar at work](https://github.gatech.edu/tcurley6/DRYAD/blob/master/rand/progressbar.gif)

### (090618)[TC]
* Finsihed what I believe is a full representation of the DRYAD model. This includes a main file by which parameters can be "tuned", and a second file that holds all helper functions (represented by different MATLAB files in Benjamin's original code). 
* Still need to annotate the helper functions in order for coders to line up the original code with the new Python code.
