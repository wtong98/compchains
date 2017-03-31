# compchains
Script for calculating basic statistics on a chain

<ul>
    <li><code>compchains_v1.py</code> is a stable script that generates the statistics, histograms (with injection value indicated by a '>' mark), and other features.</li>
    <li><code>compchains_devel.py</code> is an experimental script with new features being added to. </li>
    <li><code>compchains_old.py</code> is the original script. If all else breaks, use this one. </li>
</ul>

In the resumeproblem directory, all these scripts have been move to 'compchains.' For the sake of compatability, a soft link to compchains_v1.py remains in the directory so you won't have to modify the path to the code in your scripts

## Usage
<code>python compchains.py <u>PATH_TO_CHAIN_FILE</u> [<u>OPTIONS</u>]</code>

Example command-line with compchains_devel.py: <br>
<code>$ cd /projects/b1011/kagra/kagra_o2_lalinference/resumeproblem </code> <br>
<code>$ python ./compchains/compchains_devel.py ../386/PTMCMC.output.2895189425.h5 -sai /projects/b1011/kagra/GW150914_samples_2.xml -e 0</code>
