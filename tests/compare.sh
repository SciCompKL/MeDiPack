#
# MeDiPack, a Message Differentiation Package
#
# Copyright (C) 2017-2023 Chair for Scientific Computing (SciComp), University of Kaiserslautern-Landau
# Homepage: http://www.scicomp.uni-kl.de
# Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
#
# Lead developers: Max Sagebaum (SciComp, University of Kaiserslautern-Landau)
#
# This file is part of MeDiPack (http://www.scicomp.uni-kl.de/software/codi).
#
# MeDiPack is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# MeDiPack is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU
# Lesser General Public License along with MeDiPack.
# If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Max Sagebaum, Tim Albring (SciComp, University of Kaiserslautern-Landau)
#

#!/bin/bash

# compare.sh - compare files and dsiplays if they match
#
# Author: Max Sagebaum <max.sagebaum@scicomp.uni-kl.de>
# Date:   2015-04-22
# Category: File Comparison, TXT

# Colored output for 'ok' and 'failure'
ok="$(tput setaf 2)OK$(tput sgr 0)"
failure="$(tput setaf 1)FAILURE$(tput sgr 0)"

Usage () {
    echo >&2 "$0 - compare files with a base file and print out if they match
usage: $0 [-n name] -b baseFile files ..."

    exit 0
}

baseFileName=
testName=
while getopts n:b: opt
do
    case "$opt" in
      n)  testName=" $OPTARG";;
      b)  baseFileName="$OPTARG";;
      \?)   # unknown flag
          Usage;;
    esac
done
shift `expr $OPTIND - 1`

if [ -z $baseFileName ];
then Usage;
fi;

# arguments have been read now iterate over the files and compare them
res=
if [[ 0 == $# ]];
then res+=" $failure no drivers run this test.";
else
    while [ $# -gt 0 ]
    do
        res+=" "
        [[ $1 =~ _([^/]+)\.out ]] &&
            res+=${BASH_REMATCH[1]}:
        cmp $baseFileName $1
        if [ $? -eq 0 ];
        then res+=$ok;
        else res+=$failure;
        fi;
        shift
    done
fi;
echo Test$testName:$res
