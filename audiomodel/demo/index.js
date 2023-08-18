/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/*
- show main page and redirect
- show progress for recording and training
- add requirements to classes
-
*/

import * as tf from '@tensorflow/tfjs';
import Plotly from 'plotly.js-dist';

import * as SpeechCommands from '@tensorflow-models/speech-commands';

import {DatasetViz, removeNonFixedChildrenFromWordDiv} from './dataset-vis';
import {hideCandidateWords, logToStatusDisplay, plotPredictions, plotSpectrogram, populateCandidateWords, showCandidateWords} from './ui';

// import { Storage } from '@google-cloud/storage';
// import fs from 'fs.promises';

const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');
const predictionCanvas = document.getElementById('prediction-canvas');

const probaThresholdInput = document.getElementById('proba-threshold');
const epochsInput = document.getElementById('epochs');
const fineTuningEpochsInput = document.getElementById('fine-tuning-epochs');

const datasetIOButton = document.getElementById('dataset-io');
const datasetIOInnerDiv = document.getElementById('dataset-io-inner');
const downloadAsFileButton = document.getElementById('download-dataset');
const datasetFileInput = document.getElementById('dataset-file-input');
const uploadFilesButton = document.getElementById('upload-dataset');

const evalModelOnDatasetButton =
    document.getElementById('eval-model-on-dataset');
const evalResultsSpan = document.getElementById('eval-results');

const modelIOButton = document.getElementById('model-io');
const transferModelSaveLoadInnerDiv =
    document.getElementById('transfer-model-save-load-inner');
const loadTransferModelButton = document.getElementById('load-transfer-model');
const saveTransferModelButton = document.getElementById('save-transfer-model');
const savedTransferModelsSelect =
    document.getElementById('saved-transfer-models');
const deleteTransferModelButton =
    document.getElementById('delete-transfer-model');

const BACKGROUND_NOISE_TAG = "Background Noise";//SpeechCommands.BACKGROUND_NOISE_TAG;

/**
 * Transfer learning-related UI componenets.
 */
const transferModelNameInput = document.getElementById('transfer-model-name');
const durationMultiplierSelect = document.getElementById('duration-multiplier');
const enterLearnWordsButton = document.getElementById('enter-learn-words');
const buildLearnWordsButton = document.getElementById('build-learn-words');
const includeTimeDomainWaveformCheckbox =
    document.getElementById('include-audio-waveform');
const collectButtonsHeaderDiv = document.getElementById('collect-words-header');
const startTransferLearnButton =
    document.getElementById('start-transfer-learn');

const XFER_MODEL_NAME = 'xfer-model';

// Minimum required number of examples per class for transfer learning.
const MIN_EXAMPLES_PER_CLASS = 8;

let recognizer;
let transferWords = [];
let transferRecognizer;
let transferDurationMultiplier;

(async function() {
  logToStatusDisplay('Creating recognizer...');
  recognizer = SpeechCommands.create('BROWSER_FFT');

  await populateSavedTransferModelsSelect();

  // Make sure the tf.Model is loaded through HTTP. If this is not
  // called here, the tf.Model will be loaded the first time
  // `listen()` is called.
  recognizer.ensureModelLoaded()
      .then(() => {
        startButton.disabled = false;
        enterLearnWordsButton.disabled = false;

        transferModelNameInput.value = `model-${getDateString()}`;

        logToStatusDisplay('Model loaded.');

        const params = recognizer.params();
        logToStatusDisplay(`sampleRateHz: ${params.sampleRateHz}`);
        logToStatusDisplay(`fftSize: ${params.fftSize}`);
        logToStatusDisplay(
            `spectrogramDurationMillis: ` +
            `${params.spectrogramDurationMillis.toFixed(2)}`);
        logToStatusDisplay(
            `tf.Model input shape: ` +
            `${JSON.stringify(recognizer.modelInputShape())}`);
      })
      .catch(err => {
        logToStatusDisplay(
            'Failed to load model for recognizer: ' + err.message);
      });
})();

startButton.addEventListener('click', () => {
  const activeRecognizer =
      transferRecognizer == null ? recognizer : transferRecognizer;
  populateCandidateWords(activeRecognizer.wordLabels());

  const suppressionTimeMillis = 1000;
  activeRecognizer
      .listen(
          result => {
            plotPredictions(
                predictionCanvas, activeRecognizer.wordLabels(), result.scores,
                3, suppressionTimeMillis);
          },
          {
            includeSpectrogram: true,
            suppressionTimeMillis,
            probabilityThreshold: Number.parseFloat(probaThresholdInput.value)
          })
      .then(() => {
        startButton.disabled = true;
        stopButton.disabled = false;
        showCandidateWords();
        logToStatusDisplay('Streaming recognition started.');
      })
      .catch(err => {
        logToStatusDisplay(
            'ERROR: Failed to start streaming display: ' + err.message);
      });
});

stopButton.addEventListener('click', () => {
  const activeRecognizer =
      transferRecognizer == null ? recognizer : transferRecognizer;
  activeRecognizer.stopListening()
      .then(() => {
        startButton.disabled = false;
        stopButton.disabled = true;
        hideCandidateWords();
        logToStatusDisplay('Streaming recognition stopped.');
      })
      .catch(err => {
        logToStatusDisplay(
            'ERROR: Failed to stop streaming display: ' + err.message);
      });
});

/**
 * Transfer learning logic.
 */

/** Scroll to the bottom of the page */
function scrollToPageBottom() {
  const scrollingElement = (document.scrollingElement || document.body);
  scrollingElement.scrollTop = scrollingElement.scrollHeight;
}

let collectWordButtons = {};
let datasetViz;

function createProgressBarAndIntervalJob(parentElement, durationSec) {
  const progressBar = document.createElement('progress');
  progressBar.value = 0;
  progressBar.style['width'] = `${Math.round(window.innerWidth * 0.25)}px`;
  // Update progress bar in increments.
  const intervalJob = setInterval(() => {
    progressBar.value += 0.05;
  }, durationSec * 1e3 / 20);
  parentElement.appendChild(progressBar);
  return {progressBar, intervalJob};
}

/**
 * Create div elements for transfer words.
 *
 * @param {string[]} transferWords The array of transfer words.
 * @returns {Object} An object mapping word to th div element created for it.
 */
function createWordDivs(transferWords) {
  const wordDivs = {};
  let buttonDivs = [];
  for (const word of transferWords) {

    const sections = document.getElementsByClassName("word-div-container");

    for (let section of sections) {

      if (section.children[0].children[0].value === word) {
        const button = document.createElement('button');
        button.setAttribute('isFixed', 'true');
        button.style['display'] = 'inline-block';
        button.style['vertical-align'] = 'middle';
        const span = document.createElement("span");
        span.classList.add("material-symbols-outlined");
        span.innerText = "mic";
        button.textContent = `Record Sample`;
        button.appendChild(span);
        button.classList.add("button-sample");
        button.setAttribute("word", section.children[0].children[0].value);
        section.children[1].appendChild(button);
        buttonDivs.push(button);

        button.addEventListener('click', async () => {
          disableAllCollectWordButtons();
          removeNonFixedChildrenFromWordDiv(section.children[1]);

          const collectExampleOptions = {};
          let durationSec;
          let intervalJob;
          let progressBar;

          if (word === BACKGROUND_NOISE_TAG) {
            // If the word type is background noise, display a progress bar during
            // sound collection and do not show an incrementally updating
            // spectrogram.
            // _background_noise_ examples are special, in that user can specify
            // the length of the recording (in seconds).
            collectExampleOptions.durationSec =
                Number.parseFloat("10");
            durationSec = collectExampleOptions.durationSec;

            const barAndJob = createProgressBarAndIntervalJob(section.children[1], durationSec);
            progressBar = barAndJob.progressBar;
            intervalJob = barAndJob.intervalJob;
          } else {
            // If this is not a background-noise word type and if the duration
            // multiplier is >1 (> ~1 s recoding), show an incrementally
            // updating spectrogram in real time.
            collectExampleOptions.durationMultiplier = transferDurationMultiplier;
            let tempSpectrogramData;
            const tempCanvas = document.createElement('canvas');
            tempCanvas.style['margin-left'] = '132px';
            tempCanvas.height = 50;
            section.children[1].appendChild(tempCanvas);

            collectExampleOptions.snippetDurationSec = 0.1;
            collectExampleOptions.onSnippet = async (spectrogram) => {
              if (tempSpectrogramData == null) {
                tempSpectrogramData = spectrogram.data;
              } else {
                tempSpectrogramData = SpeechCommands.utils.concatenateFloat32Arrays(
                    [tempSpectrogramData, spectrogram.data]);
              }
              plotSpectrogram(
                  tempCanvas, tempSpectrogramData, spectrogram.frameSize,
                  spectrogram.frameSize, {pixelsPerFrame: 2});
            }
          }

          collectExampleOptions.includeRawAudio =
              includeTimeDomainWaveformCheckbox.checked;
          const spectrogram =
              await transferRecognizer.collectExample(word, collectExampleOptions);


          if (intervalJob != null) {
            clearInterval(intervalJob);
          }
          if (progressBar != null) {
            section.children[1].removeChild(progressBar);
          }
          const examples = transferRecognizer.getExamples(word)
          const example = examples[examples.length - 1];
          await datasetViz.drawExample(
            section.children[1], word, spectrogram, example.example.rawAudio, example.uid);
          enableAllCollectWordButtons();
        });
      }
    }
  }

  datasetViz = new DatasetViz(
    transferRecognizer, buttonDivs, MIN_EXAMPLES_PER_CLASS,
    startTransferLearnButton, downloadAsFileButton,
    transferDurationMultiplier);
  return wordDivs;
}










function addWordDiv(word) {
  const wordDiv = document.createElement('div');
  wordDiv.classList.add('word-div');
  wordDiv.setAttribute('word', word);
  const contentDiv = document.createElement('div');
  contentDiv.classList.add('word-div-container');
  const headerDiv = document.createElement('div');
  headerDiv.classList.add("word-div-container-header");
  const footerDiv = document.createElement('div');
  footerDiv.classList.add("word-div-container-footer");
  const displayWord = word;
  const classInputField = document.createElement("input");
  classInputField.defaultValue= `${displayWord}`;
  classInputField.classList.add('word-input-label');
  classInputField.style.width = `${word.length}ch`;
  classInputField.addEventListener("input", () => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    context.font = "20px Google Sans";
    const metrics = context.measureText(classInputField.value);
    classInputField.style.width = `${metrics.width}px`;
  });
  headerDiv.appendChild(classInputField);
  const spanItem = document.createElement("span");
  const editButton = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  spanItem.classList.add("edit-button");
  editButton.setAttribute("tabIndex", "0");
  editButton.setAttribute("viewBox", "0 0 19 18");
  editButton.setAttribute("fill", "none");
  editButton.setAttribute("width", 19);
  editButton.setAttribute("height", 18);
  editButton.setAttribute("aria-label", "Edit");
  const svgPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
  svgPath.setAttribute("fill-rule", "evenodd");
  svgPath.setAttribute("clip-rule", "evenodd");
  svgPath.setAttribute("d", "M16.06 0.590005L17.41 1.94C18.2 2.72 18.2 3.99 17.41 4.77L14.64 7.54L4.18 18H0V13.82L10.4 3.41L13.23 0.590005C14.01 -0.189995 15.28 -0.189995 16.06 0.590005ZM2 16L3.41 16.06L13.23 6.23005L11.82 4.82005L2 14.64V16Z");
  svgPath.setAttribute("fill", "#BDC1C6");
  editButton.appendChild(svgPath);
  headerDiv.appendChild(editButton);
  spanItem.appendChild(editButton);
  headerDiv.appendChild(spanItem);
  contentDiv.appendChild(headerDiv);
  contentDiv.appendChild(footerDiv);
  collectButtonsHeaderDiv.appendChild(contentDiv);
}

buildLearnWordsButton.addEventListener('click', () => {
  const total = document.getElementsByClassName("word-div-container");
  addWordDiv(`Class ${total.length}`);
});

enterLearnWordsButton.addEventListener('click', () => {
  const modelName = transferModelNameInput.value;
  if (modelName == null || modelName.length === 0) {
    enterLearnWordsButton.textContent = 'Need model name!';
    setTimeout(() => {
      enterLearnWordsButton.textContent = 'Add new class';
    }, 2000);
    return;
  }

  // We disable the option to upload an existing dataset from files
  // once the "Enter transfer words" button has been clicked.
  // However, the user can still load an existing dataset from
  // files first and keep appending examples to it.
  disableFileUploadControls();

  transferWords.push("Background Noise");

  //disabling inputs and making the other labels

  const containers = document.getElementsByClassName("edit-button");
  for (let span of Array.from(containers)) {
    span.parentNode.removeChild(span);
  }

  const inputs = document.getElementsByClassName("word-input-label");
  for (let i of Array.from(inputs)) {
    i.disabled = true;
    transferWords.push(`${i.value}`);
  }

  transferDurationMultiplier = durationMultiplierSelect.value;

  transferWords.sort();
  if (transferWords == null || transferWords.length <= 1) {
    logToStatusDisplay('ERROR: Invalid list of transfer words.');
    return;
  }

  transferRecognizer = recognizer.createTransfer(modelName);
  createWordDivs(transferWords);

  enterLearnWordsButton.parentNode.removeChild(enterLearnWordsButton);
  buildLearnWordsButton.style.display = "none";
  startButton.disabled = true;
});

function disableAllCollectWordButtons() {
  for (const word in collectWordButtons) {
    collectWordButtons[word].disabled = true;
  }
}

function enableAllCollectWordButtons() {
  for (const word in collectWordButtons) {
    collectWordButtons[word].disabled = false;
  }
}

function disableFileUploadControls() {
}

startTransferLearnButton.addEventListener('click', async () => {
  startTransferLearnButton.disabled = true;
  startButton.disabled = true;
  startTransferLearnButton.textContent = 'Model training starting...';
  await tf.nextFrame();

  const INITIAL_PHASE = 'initial';
  const FINE_TUNING_PHASE = 'fineTuningPhase';

  const epochs = parseInt(100);
  const fineTuningEpochs = parseInt(0);
  const trainLossValues = {};
  const valLossValues = {};
  const trainAccValues = {};
  const valAccValues = {};

  for (const phase of [INITIAL_PHASE, FINE_TUNING_PHASE]) {
    const phaseSuffix = phase === FINE_TUNING_PHASE ? ' (FT)' : '';
    const lineWidth = phase === FINE_TUNING_PHASE ? 2 : 1;
    trainLossValues[phase] = {
      x: [],
      y: [],
      name: 'train' + phaseSuffix,
      mode: 'lines',
      line: {width: lineWidth}
    };
    valLossValues[phase] = {
      x: [],
      y: [],
      name: 'val' + phaseSuffix,
      mode: 'lines',
      line: {width: lineWidth}
    };
    trainAccValues[phase] = {
      x: [],
      y: [],
      name: 'train' + phaseSuffix,
      mode: 'lines',
      line: {width: lineWidth}
    };
    valAccValues[phase] = {
      x: [],
      y: [],
      name: 'val' + phaseSuffix,
      mode: 'lines',
      line: {width: lineWidth}
    };
  }

  function plotLossAndAccuracy(epoch, loss, acc, val_loss, val_acc, phase) {
    const displayEpoch = phase === FINE_TUNING_PHASE ? (epoch + epochs) : epoch;
    trainLossValues[phase].x.push(displayEpoch);
    trainLossValues[phase].y.push(loss);
    trainAccValues[phase].x.push(displayEpoch);
    trainAccValues[phase].y.push(acc);
    valLossValues[phase].x.push(displayEpoch);
    valLossValues[phase].y.push(val_loss);
    valAccValues[phase].x.push(displayEpoch);
    valAccValues[phase].y.push(val_acc);



    scrollToPageBottom();
  }

  disableAllCollectWordButtons();
  const augmentByMixingNoiseRatio = null;
  console.log(`augmentByMixingNoiseRatio = ${augmentByMixingNoiseRatio}`);
  await transferRecognizer.train({
    epochs,
    validationSplit: 0.25,
    augmentByMixingNoiseRatio,
    callback: {
      onEpochEnd: async (epoch, logs) => {
        startTransferLearnButton.textContent = `Training model... (${(epoch / epochs * 1e2).toFixed(0)}%)`
      }
    },
    fineTuningEpochs,
    fineTuningCallback: {
      onEpochEnd: async (epoch, logs) => {
        startTransferLearnButton.textContent = `Training model (fine-tuning)... (${(epoch / fineTuningEpochs * 1e2).toFixed(0)}%)`
      }
    }
  });
  saveTransferModelButton.disabled = false;
  startButton.disabled = false;
  transferModelNameInput.value = transferRecognizer.name;
  transferModelNameInput.disabled = true;
  startTransferLearnButton.textContent = 'Model training complete.';
  transferModelNameInput.disabled = false;
  startButton.disabled = false;
});

/** Get the base name of the downloaded files based on current dataset. */
function getDateString() {
  const d = new Date();
  const year = `${d.getFullYear()}`;
  let month = `${d.getMonth() + 1}`;
  let day = `${d.getDate()}`;
  if (month.length < 2) {
    month = `0${month}`;
  }
  if (day.length < 2) {
    day = `0${day}`;
  }
  let hour = `${d.getHours()}`;
  if (hour.length < 2) {
    hour = `0${hour}`;
  }
  let minute = `${d.getMinutes()}`;
  if (minute.length < 2) {
    minute = `0${minute}`;
  }
  let second = `${d.getSeconds()}`;
  if (second.length < 2) {
    second = `0${second}`;
  }
  return `${year}-${month}-${day}T${hour}.${minute}.${second}`;
}

async function loadDatasetInTransferRecognizer(serialized) {
  const modelName = transferModelNameInput.value;
  if (modelName == null || modelName.length === 0) {
    throw new Error('Need model name!');
  }

  if (transferRecognizer == null) {
    transferRecognizer = recognizer.createTransfer(modelName);
  }
  transferRecognizer.loadExamples(serialized);
  const exampleCounts = transferRecognizer.countExamples();
  transferWords = [];
  const modelNumFrames = transferRecognizer.modelInputShape()[1];
  const durationMultipliers = [];
  for (const word in exampleCounts) {
    transferWords.push(word);
    const examples = transferRecognizer.getExamples(word);
    for (const example of examples) {
      const spectrogram = example.example.spectrogram;
      // Ignore _background_noise_ examples when determining the duration
      // multiplier of the dataset.
      if (word !== BACKGROUND_NOISE_TAG) {
        durationMultipliers.push(Math.round(
            spectrogram.data.length / spectrogram.frameSize / modelNumFrames));
      }
    }
  }
  transferWords.sort();

  // Determine the transferDurationMultiplier value from the dataset.
  transferDurationMultiplier =
      durationMultipliers.length > 0 ? Math.max(...durationMultipliers) : 1;
  console.log(
      `Deteremined transferDurationMultiplier from uploaded ` +
      `dataset: ${transferDurationMultiplier}`);

  createWordDivs(transferWords);
  datasetViz.redrawAll();
}

async function populateSavedTransferModelsSelect() {
  const savedModelKeys = await SpeechCommands.listSavedTransferModels();
  while (savedTransferModelsSelect.firstChild) {
    savedTransferModelsSelect.removeChild(savedTransferModelsSelect.firstChild);
  }
  if (savedModelKeys.length > 0) {
    for (const key of savedModelKeys) {
      const option = document.createElement('option');
      option.textContent = key;
      option.id = key;
      savedTransferModelsSelect.appendChild(option);
    }
  }
}

async function saveMetadata() {
  let classList = []
  classList.push(document.getElementById("bg-noise").value)
  let classLabels = document.getElementsByClassName("word-input-label")
  for (let i = 0; i < classLabels.length; i++) {
    classList.push(classLabels[i].value)
  }
  let metadata = JSON.stringify({"wordLabels": classList})
  console.log(metadata)
  var a = document.createElement("a");
  var file = new Blob([metadata], {type: "application/json"});
  a.href = URL.createObjectURL(file);
  a.download = "metadata.json";
  a.click();

}

saveTransferModelButton.addEventListener('click', async () => {
  await transferRecognizer.save('downloads://test');
  await saveMetadata();
  await populateSavedTransferModelsSelect();
  saveTransferModelButton.textContent = 'Model saved!';
  saveTransferModelButton.disabled = true;
});

