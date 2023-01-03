FROM ubuntu:jammy

# install android tools
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    ca-certificates git curl wget unzip default-jdk pulseaudio
# apt-get install qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils

# manually install Android cmdline-tools
RUN mkdir -p tmp/
RUN wget -O tmp/android_cli.zip \
    https://dl.google.com/android/repository/commandlinetools-linux-9123335_latest.zip \
    && unzip tmp/android_cli.zip -d tmp/ \
    && rm -rf tmp/android_cli \
    && mkdir -p tmp/latest/ \
    && mv tmp/cmdline-tools/* tmp/latest/ \
    && mv tmp/latest/ tmp/cmdline-tools/

# install android system images and emulator
RUN yes | tmp/cmdline-tools/latest/bin/sdkmanager --licenses \
    && tmp/cmdline-tools/latest/bin/sdkmanager --install "system-images;android-32;google_apis;x86_64" "platforms;android-32" "platform-tools"

RUN echo no | tmp/cmdline-tools/latest/bin/avdmanager create avd -f -n CogEnv -k "system-images;android-32;google_apis;x86_64"


# Download and install mamba
RUN curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh
RUN sh ./Mambaforge-$(uname)-$(uname -m).sh -b
RUN rm -rf ./Mambaforge-$(uname)-$(uname -m).sh
RUN ~/mambaforge/condabin/conda init
RUN ~/mambaforge/condabin/mamba init
