FROM node:18

RUN mkdir -p /usr/modeldemo/app

WORKDIR /usr/modeldemo/app

COPY . /usr/modeldemo/app

WORKDIR /usr/modeldemo/app/audiomodel

RUN yarn install && yarn build

WORKDIR /usr/modeldemo/app/audiomodel/demo

RUN yarn install

RUN yarn link-local

RUN yarn build

RUN adduser --disabled-password myuser

USER myuser

CMD node server.js 0.0.0.0 $PORT