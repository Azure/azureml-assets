#!/bin/bash

function add_microsoft_header() {
    # put microsoft header in NOTICE.txt
    echo '----------------------------------------------------------------------------------
    THIRD PARTY SOFTWARE NOTICES AND INFORMATION
    Do Not Translate or Localize

    This software incorporates material from third parties. Microsoft
    makes certain open source code available at
    http://3rdpartysource.microsoft.com, or you may send a check or money
    order for US $5.00, including the product name, the open source
    component name, and version number, to:

    Source Code Compliance Team
    Microsoft Corporation
    One Microsoft Way
    Redmond, WA 98052
    USA

    Notwithstanding any other terms, you may reverse engineer this
    software to the extent required to debug changes to any libraries
    licensed under the GNU Lesser General Public License for your own use.

    ----------------------------------------------------------------------------------
    ' > NOTICE.txt
}

function add_miniconda_license() {
    # append header "==> Miniconda <==" and the miniconda license to NOTICE.txt
    echo "==> Miniconda <==" >> NOTICE.txt
    miniconda_license=/opt/miniconda/LICENSE.txt
    cat $miniconda_license >> NOTICE.txt
    echo "" >> NOTICE.txt
}

function add_other_licenses() {
    # for each file under /opt/miniconda whose name contains "LICENSE"
    # excluding the miniconda license we already added
    #   1. add header "==> package_name-package_version <==" above each license
    #   2. append header + license to NOTICE.txt
    find .  -regex "./opt/miniconda/.*LICENSE.*" \
            -not -path .$miniconda_license \
            -exec tail -n +1 {} + \
        | sed "s/.*\/lib\//==> /" \
        | sed "s/.*site-packages\//==> /" \
        | sed "s/\/LICENSE.*/ <==/" \
        | sed "s/\.dist-info//" \
        | sed "s/conda\/_vendor\///" \
        >> NOTICE.txt
    echo "" >> NOTICE.txt
}

function add_copyright_files() {
    # for each file under /usr/share/doc ending in "copyright",
    #   1. add header "==> package_name-package_version <==" above each copyright
    #   2. append header + copyright to NOTICE.txt
    find /usr/share/doc -regex ".*copyright" \
                        -exec tail -n +1 {} + \
        | sed 's/==> \/usr\/share\/doc\//==> /' \
        | sed 's/\/copyright <==/ <==/' \
        >> NOTICE.txt
    echo "" >> NOTICE.txt
}

add_microsoft_header
add_miniconda_license
add_other_licenses
add_copyright_files

