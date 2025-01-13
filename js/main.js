window.onload = main;

function main() {
    tree = new DecisionTree(1000, 1000);
    tree.fit();

    document.getElementById('numPoints').addEventListener('input', function () {
        document.getElementById('numPointsOutput').textContent = `Number of points: ${this.value}`
    });

    document.getElementById('noise').addEventListener('input', function () {
        document.getElementById('noiseOutput').textContent = `Noise: ${this.value}`
    });

    document.getElementById('classAmount').addEventListener('input', function () {
        document.getElementById('classAmountOutput').textContent = `Number of classes: ${this.value}`
    });

    document.getElementById('maxDepth').addEventListener('input', function () {
        document.getElementById('maxDepthOutput').textContent = `Max depth: ${this.value}`
    });

    document.getElementById('minSamplesSplit').addEventListener('input', function () {
        document.getElementById('minSamplesSplitOutput').textContent = `Min samples split: ${this.value}`
    });

    document.getElementById('minSamplesLeaf').addEventListener('input', function () {
        document.getElementById('minSamplesLeafOutput').textContent = `Min samples leaf: ${this.value}`
    });

    document.getElementById('genSpeed').addEventListener('input', function () {
        document.getElementById('genSpeedOutput').textContent = `Generation speed: ${this.value}`
    });
}


class State {
    constructor() {
        this.state = {};
    }
}

function inBounds(p, bounds) {
    return p[0] > bounds[0][0] && p[1] > bounds[1][0] &&
        p[0] <= bounds[0][1] && p[1] <= bounds[1][1];
}




class DecisionTree {
    constructor(width, height) {
        this.tree = null;
        this.width = width;
        this.height = height;
        this.dataset = new DataSet(this, width, height);

        document.getElementById('fitBtn').addEventListener('click', () => {
            this.genSpeed = parseInt(document.getElementById('genSpeed').value);
            this.fit();
        });
        this.genSpeed = parseInt(document.getElementById('genSpeed').value);
    }

    fit() {
        this.root = null;
        this.nodeMap = {};
        this.nodeCount = 0;
        this.maxDepth = parseInt(document.getElementById('maxDepth').value);
        this.minSamplesSplit = parseInt(document.getElementById('minSamplesSplit').value);
        this.minSamplesLeaf = parseInt(document.getElementById('minSamplesLeaf').value);

        const { trainX, trainY, testX, testY } = this.splitDataset(this.dataset.X, this.dataset.y, 0.8);
        this.tree = { id: this.nodeCount++, bounds: [[0, this.width], [0, this.height]], label: this.getMostCommonLabel(trainY) }
        this.tree.depth = 0;
        this.nodeMap[this.tree.id] = this.tree;

        const nodes = []
        nodes.push(this.tree)

        const improve = () => {
            const node = nodes.pop();

            const tX = [], tY = [];
            for (let i = 0; i < trainX.length; i++) {
                if (inBounds(trainX[i], node.bounds)) {
                    tX.push(trainX[i]);
                    tY.push(trainY[i]);
                }
            }

            this.buildNode(node, tX, tY);
            if (node.left != undefined) {
                nodes.push(node.left);
            }
            if (node.right != undefined) {
                nodes.push(node.right);
            }
            const predictions = this.predict(testX);
            const accuracy = Math.round(this.calculateAccuracy(predictions, testY) * 1000) / 1000;
            const f1Score = Math.round(this.calculateF1Score(predictions, testY) * 1000) / 1000;

            document.getElementById('accuracy').textContent = `${accuracy}`;
            document.getElementById('f1Score').textContent = `${f1Score}`;

            this.showGraph();
            this.dataset.show();

            if (node.left === undefined && node.right === undefined && nodes.length > 0) {
                setTimeout(improve, 0);
            } else if (nodes.length > 0) {
                setTimeout(improve, this.genSpeed);
            }
        }
        improve();

        // while (nodes.length > 0) {
        //     const node = nodes.pop();

        //     const tX = [], tY = [];
        //     for (let i = 0; i < trainX.length; i++) {
        //         if (inBounds(trainX[i], node.bounds)) {
        //             tX.push(trainX[i]);
        //             tY.push(trainY[i]);
        //         }
        //     }

        //     this.buildNode(node, tX, tY);
        //     if (node.left != undefined) {
        //         nodes.push(node.left);
        //     }
        //     if (node.right != undefined) {
        //         nodes.push(node.right);
        //     }
        //     this.showGraph();
        //     this.dataset.show();
        // }

        // const predictions = this.predict(testX);
        // const accuracy = Math.round(this.calculateAccuracy(predictions, testY) * 1000) / 1000;
        // const f1Score = Math.round(this.calculateF1Score(predictions, testY) * 1000) / 1000;

        // document.getElementById('accuracy').textContent = `${accuracy}`;
        // document.getElementById('f1Score').textContent = `${f1Score}`;

        // this.showGraph();
        // this.dataset.show();
    }

    nodeStats(nodeId) {
        if (nodeId === undefined) {
            const canvas = document.getElementById('nodeCanvas');
            const ctx = canvas.getContext('2d');

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            return;
        }
        const node = this.nodeMap[nodeId];
        const xInNode = [];
        const yInNode = [];
        const colorCounts = {};

        for (let i = 0; i < this.dataset.X.length; i++) {
            const x = this.dataset.X[i];
            if (x[0] > node.bounds[0][0] && x[0] < node.bounds[0][1] && x[1] > node.bounds[1][0] && x[1] < node.bounds[1][1]) {
                const y = this.dataset.y[i];
                yInNode.push(y);
                xInNode.push(x);
                colorCounts[y] = (colorCounts[y] || 0) + 1;
            }
        }

        const predictedClass = node.label

        const bestSplit = this.getBestSplit(xInNode, yInNode);
        const bestGain = bestSplit.bestGain || 0;

        const canvas = document.getElementById('nodeCanvas');
        const ctx = canvas.getContext('2d');

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const colors = Object.keys(colorCounts);

        const centerX = 30;
        let y = 50;
        const fixedRadius = 20;
        const textOffset = 10;

        colors.forEach(color => {
            const count = colorCounts[color];

            ctx.beginPath();
            ctx.arc(centerX, y, fixedRadius, 0, Math.PI * 2, false);
            ctx.fillStyle = this.dataset.classColors[color];
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = 'black';
            ctx.font = '14px Arial';
            ctx.fillText(`${color}: ${count}`, centerX + fixedRadius + textOffset, y + 5);

            y += 2 * fixedRadius + 20;
        });

        ctx.fillStyle = 'black';
        ctx.font = '16px Arial';
        if (predictedClass !== undefined) ctx.fillText(`Predicted Class: ${predictedClass}`, 10, y);
        if (predictedClass === undefined) ctx.fillText(`Best Gain: ${bestGain.toFixed(2)}`, 10, y);

        return;
    }

    calculateAccuracy(predictions, actual) {
        let correct = 0;
        for (let i = 0; i < predictions.length; i++) {
            if (predictions[i] === actual[i]) correct++;
        }
        return correct / actual.length;
    }

    calculateF1Score(predictions, actual) {
        let tp = 0, fp = 0, fn = 0;
        for (let i = 0; i < predictions.length; i++) {
            if (predictions[i] === 1 && actual[i] === 1) {
                tp++;
            } else if (predictions[i] === 1 && actual[i] === 0) {
                fp++;
            } else if (predictions[i] === 0 && actual[i] === 1) {
                fn++;
            }
        }
        const precision = tp / (tp + fp);
        if (tp + fp === 0) return tp === 0 ? 1 : 0;
        const recall = tp / (tp + fn);
        return 2 * (precision * recall) / (precision + recall);
    }

    splitDataset(X, y, trainSize = 0.8) {
        const dataSize = X.length;
        const trainCount = Math.floor(dataSize * trainSize + 1);


        const indices = Array.from(Array(dataSize).keys());
        indices.sort(() => Math.random() - 0.5);

        const trainIndices = indices.slice(0, trainCount);
        const testIndices = indices.slice(trainCount);

        return {
            trainX: trainIndices.map(i => X[i]),
            trainY: trainIndices.map(i => y[i]),
            testX: testIndices.map(i => X[i]),
            testY: testIndices.map(i => y[i]),
        };
    }

    showGraph() {
        const { nodes, edges } = this.generateVisData(this.tree);
        const container = document.getElementById("visualization");
        const data = {
            nodes: new vis.DataSet(nodes),
            edges: new vis.DataSet(edges)
        };
        const options = {
            layout: {
                hierarchical: {
                    direction: 'UD',
                    sortMethod: 'directed'
                }
            },
            physics: true,
        };
        const network = new vis.Network(container, data, options);
        const _this = this;
        network.on('click', function (params) {
            const nodeId = this.getNodeAt(params.pointer.DOM);

            _this.dataset.selectedNode = _this.nodeMap[nodeId];
            _this.dataset.show();
            _this.nodeStats(nodeId);
        });
    }

    generateVisData(tree, nodes = [], edges = []) {
        let nodeLabel, nodeShape, nodeColor;

        if (tree.label != undefined) {
            nodeLabel = `${tree.label}`;
            nodeShape = 'circle';
            nodeColor = this.dataset.classColors[tree.label] || '#dddddd';
        } else {
            nodeLabel = `${tree.featureIndex === 0 ? 'X' : 'Y'} <= ${Math.round(tree.threshold)}`;
            nodeShape = 'box';
            nodeColor = '#ffffff';
            if (tree.left != undefined) {
                edges.push({ from: tree.id, to: tree.left.id, label: "yes" });
                this.generateVisData(tree.left, nodes, edges);
            }
            if (tree.right != undefined) {
                edges.push({ from: tree.id, to: tree.right.id, label: "no" });
                this.generateVisData(tree.right, nodes, edges);
            }
        }

        nodes.push({
            id: tree.id,
            label: nodeLabel,
            shape: nodeShape,
            color: {
                background: nodeColor,
                border: '#000000',
                highlight: {
                    background: nodeColor,
                    border: '#000000'
                }
            },
            font: {
                color: '#000000',
                size: 14,
                align: 'middle'
            }
        });

        return { nodes, edges };
    }

    buildTree(cur, X, y, parent = null, depth = 0, bounds = undefined) {
        this.showGraph();
        // this.dataset.show();
        cur.parent = parent

        if (bounds === undefined) {
            bounds = [
                [0, this.dataset.width],
                [0, this.dataset.height]
            ];
        }

        cur.id = this.nodeCount++;
        cur.bounds = bounds;
        this.nodeMap[cur.id] = cur;

        const nSamples = X.length;
        const nFeatures = X[0].length;
        if (nSamples < this.minSamplesSplit || depth === this.maxDepth) {
            return;
        }

        const { bestSplit, bestGain } = this.getBestSplit(X, y);
        if (!bestSplit || bestGain === 0) {
            return;
        }

        cur.label = undefined

        const { featureIndex, threshold, leftMask, rightMask } = bestSplit;
        cur.featureIndex = featureIndex;
        cur.threshold = threshold;

        const leftBounds = JSON.parse(JSON.stringify(bounds));
        leftBounds[featureIndex][1] = Math.min(bounds[featureIndex][1], threshold);
        const rightBounds = JSON.parse(JSON.stringify(bounds));
        rightBounds[featureIndex][0] = Math.max(bounds[featureIndex][0], threshold);

        var leftTree = {}
        this.buildTree(leftTree, this.filterX(X, leftMask), this.filterY(y, leftMask), cur, depth + 1, leftBounds);
        var rightTree = {}
        this.buildTree(rightTree, this.filterX(X, rightMask), this.filterY(y, rightMask), cur, depth + 1, rightBounds);

        cur.left = leftTree;
        cur.right = rightTree;
        return;
    }

    buildNode(cur, X, y) {
        const nSamples = X.length;
        const nFeatures = X[0].length;
        if (nSamples < this.minSamplesSplit || cur.depth === this.maxDepth) {
            return;
        }

        const { bestSplit, bestGain } = this.getBestSplit(X, y);
        if (!bestSplit || bestGain === 0) {
            return;
        }

        cur.label = undefined
        const { featureIndex, threshold, leftMask, rightMask } = bestSplit;
        cur.featureIndex = featureIndex;
        cur.threshold = threshold;

        const bounds = cur.bounds;
        const leftBounds = JSON.parse(JSON.stringify(bounds));
        leftBounds[featureIndex][1] = Math.min(bounds[featureIndex][1], threshold);
        const rightBounds = JSON.parse(JSON.stringify(bounds));
        rightBounds[featureIndex][0] = Math.max(bounds[featureIndex][0], threshold);

        const leftId = this.nodeCount++;
        cur.left = { id: leftId, label: this.getMostCommonLabel(this.filterY(y, leftMask)), depth: cur.depth + 1, bounds: leftBounds };
        const rightId = this.nodeCount++;
        cur.right = { id: rightId, label: this.getMostCommonLabel(this.filterY(y, rightMask)), depth: cur.depth + 1, bounds: rightBounds };

        this.nodeMap[leftId] = cur.left;
        this.nodeMap[rightId] = cur.right;

        return;
    }

    getBestSplit(X, y) {
        const nFeatures = X[0].length;
        let bestGain = 0;
        let bestSplit = null;

        for (let featureIndex = 0; featureIndex < nFeatures; featureIndex++) {
            const thresholds = [...new Set(X.map(row => row[featureIndex]))];
            for (let i = 0; i < thresholds.length - 1; i++) {
                thresholds[i] = (thresholds[i] + thresholds[i + 1]) / 2;
            }

            thresholds.forEach(threshold => {
                const leftMask = X.map(row => row[featureIndex] <= threshold);
                const rightMask = X.map(row => row[featureIndex] > threshold);

                if (leftMask.reduce((sum, val) => sum + val, 0) < this.minSamplesLeaf ||
                    rightMask.reduce((sum, val) => sum + val, 0) < this.minSamplesLeaf) {
                    return;
                }

                const gain = this.giniGain(y, leftMask, rightMask);
                if (gain > bestGain) {
                    bestGain = gain;
                    bestSplit = { featureIndex, threshold, leftMask, rightMask };
                }
            });
        }
        return { bestSplit, bestGain };
    }

    giniGain(y, leftMask, rightMask) {
        const yGini = this.gini(y);
        const nLeft = leftMask.reduce((sum, val) => sum + val, 0);
        const nRight = rightMask.reduce((sum, val) => sum + val, 0);
        const nTotal = y.length;
        const leftGini = this.gini(this.filterY(y, leftMask));
        const rightGini = this.gini(this.filterY(y, rightMask));
        const newYGini = (nLeft / nTotal) * leftGini + (nRight / nTotal) * rightGini;
        return yGini - newYGini;
    }

    filterX(X, mask) {
        return X.filter((_, i) => mask[i]);
    }

    filterY(y, mask) {
        return y.filter((_, i) => mask[i]);
    }

    gini(y) {
        const counter = y.reduce((acc, value) => {
            acc[value] = (acc[value] || 0) + 1;
            return acc;
        }, {});
        const proba = Object.values(counter).map(count => count / y.length);
        return 1 - proba.reduce((sum, p) => sum + p ** 2, 0);
    }

    getMostCommonLabel(y) {
        const counter = y.reduce((acc, value) => {
            acc[value] = (acc[value] || 0) + 1;
            return acc;
        }, {});
        return parseInt(Object.keys(counter).reduce((a, b) => counter[a] > counter[b] ? a : b));
    }

    predict(X) {
        return X.map(x => this.predictOne(x, this.tree));
    }

    predictOne(x, tree) {
        if (tree.label != undefined) {
            return tree.label;
        }
        const { featureIndex, threshold, left, right } = tree;
        if (x[featureIndex] <= threshold) {
            return this.predictOne(x, left);
        } else {
            return this.predictOne(x, right);
        }
    }

    getDepth() {
        return this.getDepthHelper(this.tree);
    }

    getDepthHelper(tree) {
        if (tree.label != undefined) {
            return 0;
        }
        const leftDepth = this.getDepthHelper(tree.left);
        const rightDepth = this.getDepthHelper(tree.right);
        return Math.max(leftDepth, rightDepth) + 1;
    }
}

class DataSet {
    constructor(tree, width, height) {
        this.tree = tree;
        const { X, y } = generateClusters(40, width, height, 3, 30);
        this.X = X;
        this.y = y;
        this.width = width;
        this.height = height;
        this.classColors = {
            0: 'red',
            1: 'lightblue',
            2: 'green',
            3: 'yellow',
            4: 'orange',
        }

        document.getElementById('clearBtn').addEventListener('click', () => {
            this.X = [[this.width / 2, this.height / 2]];
            this.y = [0];
            this.tree.fit();
        });

        document.getElementById('generateBtn').addEventListener('click', () => {
            const dataSelector = document.getElementById('dataSelector');
            let X, y;

            const noise = parseInt(document.getElementById('noise').value);
            const numPoints = parseInt(document.getElementById('numPoints').value);
            const classAmount = parseInt(document.getElementById('classAmount').value);

            if (dataSelector.value === 'random') {
                ({ X, y } = generateRandomData(numPoints, 600, 600, classAmount, noise));
            } else if (dataSelector.value === 'spiral') {
                ({ X, y } = generateSpiralData(numPoints, 600, 600, classAmount, noise));
            } else if (dataSelector.value === 'concentricCircles') {
                ({ X, y } = generateConcentricCircles(numPoints, 600, 600, classAmount, noise));
            } else if (dataSelector.value === 'parallelLines') {
                ({ X, y } = generateParallelLines(numPoints, 600, 600, classAmount, noise));
            } else if (dataSelector.value === 'waveLines') {
                ({ X, y } = generateWaveLines(numPoints, 600, 600, classAmount, noise));
            } else if (dataSelector.value === 'clusters') {
                ({ X, y } = generateClusters(numPoints, 600, 600, classAmount, noise));
            }

            this.X = X;
            this.y = y;
            this.tree.genSpeed = document.getElementById('genSpeed').value;
            this.tree.fit();
        });

        const canvas = document.getElementById('datacanvas');
        canvas.addEventListener('click', (event) => {
            const rect = event.target.getBoundingClientRect();

            const x = (event.clientX - rect.left) / (rect.right - rect.left) * canvas.width;
            const y = (event.clientY - rect.top) / (rect.bottom - rect.top) * canvas.height;
            const selectClassBtn = document.getElementById('selectClassBtn');
            const newLabel = parseInt(selectClassBtn.dataset.choosed, 10);
            this.addPoint(x, y, newLabel);
            this.tree.genSpeed = 0;
            this.tree.fit();
        });

        const selectClassBtn = document.getElementById('selectClassBtn');
        selectClassBtn.addEventListener("click", () => {
            console.log(selectClassBtn.dataset.choosed);
            selectClassBtn.dataset.choosed++;
            selectClassBtn.dataset.choosed %= 3;
            selectClassBtn.style.backgroundColor = this.classColors[selectClassBtn.dataset.choosed];
        });

        this.show();
    }

    show() {
        const canvas = document.getElementById('datacanvas');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        this.y.forEach((label, index) => {
            if (!this.classColors[label]) {
                this.classColors[label] = this.getRandomColor();
            }
            const [x, y] = this.X[index];
            ctx.fillStyle = this.classColors[label];
            ctx.beginPath();
            var radius = 5;

            if (this.selectedNode != undefined &&
                x >= this.selectedNode.bounds[0][0] &&
                x <= this.selectedNode.bounds[0][1] &&
                y >= this.selectedNode.bounds[1][0] &&
                y <= this.selectedNode.bounds[1][1]
            ) {
                radius = 8;
            }
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fill();
        });

        for (var nodeId in this.tree.nodeMap) {
            const node = this.tree.nodeMap[nodeId];
            const margin = 2;

            if (node.label != undefined) {
                ctx.strokeStyle = this.classColors[node.label] || '#000';
                ctx.beginPath();
                ctx.lineWidth = 3;
                ctx.strokeRect(
                    node.bounds[0][0] + margin,
                    node.bounds[1][0] + margin,
                    node.bounds[0][1] - node.bounds[0][0] - 2 * margin,
                    node.bounds[1][1] - node.bounds[1][0] - 2 * margin
                );
            }

            if (this.selectedNode != undefined && this.selectedNode.id == node.id) {
                ctx.fillStyle = 'rgba(0, 0, 255, 0.1)';
                ctx.fillRect(
                    node.bounds[0][0] + margin,
                    node.bounds[1][0] + margin,
                    node.bounds[0][1] - node.bounds[0][0] - 2 * margin,
                    node.bounds[1][1] - node.bounds[1][0] - 2 * margin
                );
            }
        }
    }

    getRandomColor() {
        const letters = '0123456789ABCDEF';
        let color = '#';
        for (let i = 0; i < 6; i++) {
            color += letters[Math.floor(Math.random() * 16)];
        }
        return color;
    }

    addPoint(x, y, label) {
        this.X.push([x, y]);
        this.y.push(label);
        this.show();
    }
}

function generateSpiralData(numPoints, height, width, numClasses = 2, noise = 0) {
    const X = [];
    const y = [];
    const radius = Math.min(height, width) / 2;

    for (let j = 0; j < numClasses; j++) {
        const deltaTheta = (2 * Math.PI) / numClasses;
        const thetaOffset = j * deltaTheta;

        for (let i = 0; i < numPoints; i++) {
            const r = (i / numPoints) * radius;
            const theta = thetaOffset + (i / numPoints) * 4 * Math.PI;

            const x = width / 2 + r * Math.cos(theta) + Math.random() * noise;
            const yCoord = height / 2 + r * Math.sin(theta) + Math.random() * noise;

            X.push([x, yCoord]);
            y.push(j);
        }
    }
    return { X, y };
}

function generateConcentricCircles(numPoints, height, width, numClasses = 2, noise = 0) {
    const X = [];
    const y = [];
    const radiusIncrement = Math.min(height, width) / (3 * numClasses);

    for (let j = 0; j < numClasses; j++) {
        const radius = radiusIncrement * (j + 1);

        for (let i = 0; i < numPoints; i++) {
            const theta = (i / numPoints) * 2 * Math.PI;

            const x = width / 2 + radius * Math.cos(theta) + Math.random() * noise;
            const yCoord = height / 2 + radius * Math.sin(theta) + Math.random() * noise;

            X.push([x, yCoord]);
            y.push(j);
        }
    }
    return { X, y };
}

function generateParallelLines(numPoints, height, width, numLines = 2, noise = 0) {
    const X = [];
    const y = [];
    const spacing = width / (numLines + 1);

    for (let j = 0; j < numLines; j++) {
        const xOffset = (j + 1) * spacing;

        for (let i = 0; i < numPoints; i++) {
            const yCoord = (i / numPoints) * height + Math.random() * noise;

            X.push([xOffset + Math.random() * noise, yCoord]);
            y.push(j);
        }
    }
    return { X, y };
}

function generateWaveLines(numPoints, height, width, numLines = 3, noise = 0) {
    const X = [];
    const y = [];
    const amplitude = height / (numLines + 1) / 1.5;
    const frequency = 2;

    for (let j = 0; j < numLines; j++) {
        const yOffset = (j + 1) * amplitude * 1.5;

        for (let i = 0; i < numPoints; i++) {
            const x = (i / numPoints) * width;
            const yCoord = yOffset + amplitude * Math.sin(frequency * (2 * Math.PI) * (i / numPoints));

            X.push([x + Math.random() * noise, yCoord + Math.random() * noise]);
            y.push(j);
        }
    }
    return { X, y };
}

function generateClusters(numPoints, height, width, numClusters, noise = 0) {
    const X = [];
    const y = [];
    const centers = [];

    for (let j = 0; j < numClusters; j++) {
        let validCenter = false;
        let centerX, centerY;
        const minDistance = Math.min(height, width) / 5;
        while (!validCenter) {
            const padding = minDistance;
            centerX = padding + Math.random() * (width - 2 * padding);
            centerY = padding + Math.random() * (height - 2 * padding);
            validCenter = true;

            for (const [existingX, existingY] of centers) {
                const distance = Math.sqrt((centerX - existingX) ** 2 + (centerY - existingY) ** 2);
                if (distance < minDistance) {
                    validCenter = false;
                    break;
                }
            }
        }

        centers.push([centerX, centerY]);

        for (let i = 0; i < numPoints; i++) {
            const x = centerX + (Math.random() - 0.5) * noise / 100.0 * width;
            const yCoord = centerY + (Math.random() - 0.5) * noise / 100.0 * height;

            X.push([x, yCoord]);
            y.push(j);
        }
    }
    return { X, y };
}

function generateRandomData(numPoints, height, width, numClasses = 2, noise = 0) {
    const X = [];
    const y = [];

    for (let i = 0; i < numPoints; i++) {
        const x = Math.random() * width + Math.random() * noise;
        const yCoord = Math.random() * height + Math.random() * noise;

        X.push([x, yCoord]);
        y.push(Math.floor(Math.random() * numClasses));
    }
    return { X, y };
}
