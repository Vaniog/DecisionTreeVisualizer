window.onload = function () {
    dataset = new Dataset([[100, 200], [200, 300], [300, 350]], [0, 0, 1]);
    dataset.classColors = {
        0: 'black',
        1: 'lightblue',
        2: 'green',
        3: 'yellow',
        4: 'orange',
    }
    tree = new DecisionTree();
    tree.classColors = dataset.classColors

    dataset.show()
    tree.fit(dataset.X, dataset.y);
    dataset.displayTreeBounds(tree)
    tree.show(id => dataset.highlightNode(tree, id));

    document.getElementById('fitBtn').addEventListener('click', () => {
        maxDepth = parseInt(document.getElementById('maxDepth').value);
        minSamplesSplit = parseInt(document.getElementById('minSamplesSplit').value);
        minSamplesLeaf = parseInt(document.getElementById('minSamplesLeaf').value);

        tree = new DecisionTree(maxDepth, minSamplesSplit, minSamplesLeaf);
        tree.classColors = dataset.classColors;
        tree.fit(dataset.X, dataset.y);
        dataset.show()
        dataset.displayTreeBounds(tree)
        tree.show(id => dataset.highlightNode(tree, id));
    });

    document.getElementById("generateBtn").addEventListener('click', () => {
        const dataSelector = document.getElementById('dataSelector');
        let X, y;

        noise = parseInt(document.getElementById('noise').value);
        numPoints = parseInt(document.getElementById('numPoints').value);
        classAmount = parseInt(document.getElementById('classAmount').value);

        if (dataSelector.value === 'clear') {
            X = [[300, 300]];
            y = [0];
        }
        if (dataSelector.value === 'spiral') {
            ({ X, y } = generateSpiralData(numPoints, 600, 600, classAmount, noise));
        } else if (dataSelector.value === 'concentricCircles') {
            ({ X, y } = generateConcentricCircles(numPoints, 600, 600, classAmount, noise));
        } else if (dataSelector.value === 'parallelLines') {
            ({ X, y } = generateParallelLines(numPoints, 600, 600, classAmount, noise));
        } else if (dataSelector.value === 'waveLines') {
            ({ X, y } = generateWaveLines(numPoints, 600, 600, classAmount, noise));
        }
        dataset = new Dataset(X, y, dataset.classColors);

        maxDepth = parseInt(document.getElementById('maxDepth').value);
        minSamplesSplit = parseInt(document.getElementById('minSamplesSplit').value);
        minSamplesLeaf = parseInt(document.getElementById('minSamplesLeaf').value);

        tree = new DecisionTree(maxDepth, minSamplesSplit, minSamplesLeaf);
        tree.classColors = dataset.classColors;
        tree.fit(dataset.X, dataset.y);
        dataset.show()
        dataset.displayTreeBounds(tree)
        tree.show(id => dataset.highlightNode(tree, id));
    });

    document.getElementById('numPoints').addEventListener('input', function () {
        document.getElementById('numPointsOutput').textContent = `Number of points: ${this.value}`
    });

    document.getElementById('noise').addEventListener('input', function () {
        document.getElementById('noiseOutput').textContent = `Noise: ${this.value}`
    });

    document.getElementById('classAmount').addEventListener('input', function () {
        document.getElementById('classAmountOutput').textContent = `Number of classes: ${this.value}`
    });
}

class DecisionTree {
    constructor(maxDepth = null, minSamplesSplit = 2, minSamplesLeaf = 1) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.tree = null;
        this.nodeId = 0;
    }

    fit(X, y) {
        this.nodeId = 0;
        this.tree = this.buildTree(X, y);
    }

    buildTree(X, y, parent = null, depth = 0) {
        const nodeId = this.nodeId++;
        const nSamples = X.length;
        const nFeatures = X[0].length;
        if (nSamples < this.minSamplesSplit || depth === this.maxDepth) {
            return { id: nodeId, label: this.getMostCommonLabel(y) };
        }

        const { bestSplit, bestGain } = this.getBestSplit(X, y);
        if (!bestSplit || bestGain === 0) {
            return { id: nodeId, label: this.getMostCommonLabel(y) };
        }

        const { featureIndex, threshold, leftMask, rightMask } = bestSplit;
        const newNode = new Node(nodeId, featureIndex, threshold, parent);
        const leftTree = this.buildTree(this.filterX(X, leftMask), this.filterY(y, leftMask), newNode, depth + 1);
        const rightTree = this.buildTree(this.filterX(X, rightMask), this.filterY(y, rightMask), newNode, depth + 1);

        newNode.left = leftTree;
        newNode.right = rightTree;
        return newNode;
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

    show(highlightCallback) {
        console.log("hi", highlightCallback);
        this.showVis(highlightCallback);
    }

    showVis(highlightCallback) {
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
        network.on('click', function (params) {
            const nodeId = this.getNodeAt(params.pointer.DOM);

            if (nodeId != undefined) {
                highlightCallback(nodeId);
            }
        });
    }

    generateVisData(tree, nodes = [], edges = []) {
        let nodeLabel, nodeShape, nodeColor;

        if (tree.label != undefined) {
            nodeLabel = `${tree.label}`;
            nodeShape = 'circle';
            nodeColor = this.classColors[tree.label] || '#dddddd';
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
}

class Node {
    constructor(id, featureIndex, threshold, parent, left = null, right = null) {
        this.id = id;
        this.featureIndex = featureIndex;
        this.threshold = threshold;
        this.parent = parent;
        this.left = left;
        this.right = right;
    }
}

class Dataset {
    constructor(X, y, classColors = {}) {
        this.X = X;
        this.y = y;
        this.classColors = classColors;
        this.highlighted = null;


        const canvas = document.getElementById('datacanvas');
        canvas.addEventListener('click', (event) => {
            const rect = event.target.getBoundingClientRect();

            const x = (event.clientX - rect.left) / (rect.right - rect.left) * canvas.width;
            const y = (event.clientY - rect.top) / (rect.bottom - rect.top) * canvas.height;
            const classSelector = document.getElementById('classSelector');
            const newLabel = parseInt(classSelector.value, 10);
            dataset.addPoint(x, y, newLabel);
        });
    }

    show(clear = true) {
        const canvas = document.getElementById('datacanvas');
        const ctx = canvas.getContext('2d');
        if (clear) ctx.clearRect(0, 0, canvas.width, canvas.height);

        this.y.forEach((label, index) => {
            if (!this.classColors[label]) {
                this.classColors[label] = this.getRandomColor();
            }
            const [x, y] = this.X[index];
            ctx.fillStyle = this.classColors[label];
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    displayTreeBounds(tree) {
        this.displayNode(tree.tree, [[0, 600], [0, 600]])
    }

    highlightNode(tree, nodeId) {
        console.log(this)
        this.highlighted = nodeId
        const canvas = document.getElementById('datacanvas');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        this.displayTreeBounds(tree)
        this.show(false)
    }

    displayNode(node, bounds) {
        if (node === undefined) return;
        console.log(node)
        if (node.label != undefined) {
            console.log(bounds);
            const canvas = document.getElementById('datacanvas');
            const ctx = canvas.getContext('2d');

            if (this.highlighted === node.id) {
                ctx.fillStyle = 'rgba(0, 0, 255, 0.1)';
                ctx.fillRect(
                    bounds[0][0] + 2,
                    bounds[1][0] + 2,
                    bounds[0][1] - bounds[0][0] - 4,
                    bounds[1][1] - bounds[1][0] - 4
                );
            }

            ctx.strokeStyle = this.classColors[node.label] || '#000';
            ctx.strokeRect(
                bounds[0][0] + 2,
                bounds[1][0] + 2,
                bounds[0][1] - bounds[0][0] - 4,
                bounds[1][1] - bounds[1][0] - 4
            );
        }


        if (node.left !== undefined) {
            const newBounds = JSON.parse(JSON.stringify(bounds));
            newBounds[node.featureIndex][1] = Math.min(newBounds[node.featureIndex][1], node.threshold);
            this.displayNode(node.left, newBounds);
        }
        if (node.right !== undefined) {
            const newBounds = JSON.parse(JSON.stringify(bounds));
            newBounds[node.featureIndex][0] = Math.max(newBounds[node.featureIndex][0], node.threshold);
            this.displayNode(node.right, newBounds);
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

function giniGain(y, masks) {
    const yGini = gini(y);
    const [leftMask, rightMask] = masks;
    const nLeft = leftMask.reduce((sum, val) => sum + val, 0);
    const nRight = rightMask.reduce((sum, val) => sum + val, 0);
    const nTotal = y.length;
    const leftGini = gini(y.filter((_, i) => leftMask[i]));
    const rightGini = gini(y.filter((_, i) => rightMask[i]));
    const newYGini = (nLeft / nTotal) * leftGini + (nRight / nTotal) * rightGini;

    return yGini - newYGini;
}

function gini(y) {
    const counter = y.reduce((acc, value) => {
        acc[value] = (acc[value] || 0) + 1;
        return acc;
    }, {});

    const proba = Object.values(counter).map(count => count / y.length);
    return 1 - proba.reduce((sum, p) => sum + p ** 2, 0);
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
    const amplitude = height / (numLines + 1);
    const frequency = 2; // цикл волны на всей ширине

    for (let j = 0; j < numLines; j++) {
        const yOffset = (j + 1) * amplitude;

        for (let i = 0; i < numPoints; i++) {
            const x = (i / numPoints) * width;
            const yCoord = yOffset + amplitude * Math.sin(frequency * (2 * Math.PI) * (i / numPoints));

            X.push([x + Math.random() * noise, yCoord + Math.random() * noise]);
            y.push(j);
        }
    }
    return { X, y };
}
