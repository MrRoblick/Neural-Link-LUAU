--!strict
local HttpService: HttpService = game:GetService('HttpService')
local random: Random = Random.new()

export type Matrix = {{number}}
type NeuralNetwork = {
	HiddenLayer: {number},
	InputLayer: {number},
	OutputLayer: {number},
	WeightHidden: Matrix,
	WeightOutput: Matrix,
	ErrOutput: {number},
	ErrHidden: {number},
	LastChangeHidden: Matrix,
	LastChangeOutput: Matrix,
	Regression: boolean,
	Rate1: number,
	Rate2: number,
	
	Feedback: (target: {number}) -> (),
	CalcError: (target: {number}) -> number,
	Train: (inputs: Matrix, targets: Matrix, iterations: number, debugging: boolean?) -> (),
	TrainMap: (inputs: {{[string]: number}}, targets: Matrix, iterations: number) -> (),
	ForwardMap: (input: {[string]: number}) -> {number},
	FeedbackMap: (target: {number}, input: {[string]: number}) -> (),
	Forward: (input: {number}) -> {number},	
}
type ModuleImpl = {
	new: (inputCount: number, hiddenCount: number, outputCount: number, regression: boolean, debugging: boolean?) -> NeuralNetwork,
	SaveNN: (NeuralNetwork: NeuralNetwork) -> string,
	LoadNN: (Model: string)->NeuralNetwork,
}
local function makeArray(length: number): {number}
	local x: {number} = {}
	for i=1, length do
		x[i] = 0
	end
	return x
end
local function sigmoid(x: number): number
	return 1.0 / (1.0 + math.exp(-x))
end
local function dsigmoid(x: number): number
	return x * (1.0 - x)
end
local function genRandomIdx(N :number): {number}
	local A: {number} = makeArray(N)
	for i = 1, N do
		A[i] = i
	end
	for i = 1, N do
		local j = i + math.floor(random:NextNumber(0, N - i))
		A[i], A[j] = A[j], A[i]
	end
	return A
end
local function makeMatrix(rows: number, colums: number, value: number): Matrix
	local matrix = {} :: Matrix
	for i = 1, rows do
		matrix[i] = {}
		for j = 1, colums do
			matrix[i][j] = value
		end
	end
	return matrix
end
local function randomMatrix(rows: number, columns: number, lower: number, upper: number): Matrix
	local matrix = {} :: Matrix
	for i = 1, rows do
		matrix[i] = {}
		for j = 1, columns do
			matrix[i][j] = random:NextNumber(lower, upper)
		end
	end
	return matrix
end


local nn = {} :: ModuleImpl
local function InitMethods(network: NeuralNetwork, debugging: boolean?)
	network.Train = function(inputs: Matrix, targets: Matrix, iteration: number): ()
		if #inputs[1] ~= #network.InputLayer then
			error("amount of input variable doesn't match")
		elseif #targets[1] ~= #network.OutputLayer then
			error("amount of output variable doesn't match")
		end

		local iter_flag: number = -1

		for i = 1, iteration do
			local idx_ary: {number} = genRandomIdx(#inputs)
			local cur_err: number = 0.0
			for j = 1, #inputs do
				network.Forward(inputs[idx_ary[j]])
				network.Feedback(targets[idx_ary[j]])
				cur_err += network.CalcError(targets[idx_ary[j]])
				if j%1000 == 0 then
					iter_flag = iter_flag ~= i and i or iter_flag
					if debugging then print(`iteration {i}th / progress {string.format("%.2f", j*100 / #inputs)}`) end
				end
			end
			if ((iteration >= 10 and i%(iteration/10) == 0) or iteration < 10) and debugging then
				print(`iteration {i}th MSE: {string.format("%.5f", cur_err / #inputs)}`)
			end
		end
		if debugging then print("done.") end
	end
	network.Feedback = function(target: {number}): ()
		for i = 1, #network.OutputLayer do
			network.ErrOutput[i] = network.OutputLayer[i] - target[i]
		end
		for i = 1, #network.HiddenLayer-1 do
			local err: number = 0.0
			for j = 1, #network.OutputLayer do
				local l: number = network.ErrOutput[j] * network.WeightOutput[j][i]
				err += network.Regression and l or l * dsigmoid(network.OutputLayer[j])
			end
			network.ErrHidden[i] = err
		end

		for i = 1, #network.OutputLayer do
			for j = 1, #network.HiddenLayer do
				local change: number = 0.0
				local delta: number = 0.0
				delta = network.Regression and network.ErrOutput[i] or network.ErrOutput[i] * dsigmoid(network.OutputLayer[i])
				change = network.Rate1*delta*network.HiddenLayer[j] + network.Rate2*network.LastChangeOutput[i][j]

				network.WeightOutput[i][j] -= change
				network.LastChangeOutput[i][j] = change

			end
		end
		for i = 1, #network.HiddenLayer-1 do
			for j = 1, #network.InputLayer do
				local delta: number = network.ErrHidden[i] * dsigmoid(network.HiddenLayer[i])
				local change: number = network.Rate1 * delta * network.InputLayer[j] + network.Rate2*network.LastChangeHidden[i][j]
				network.WeightHidden[i][j] -= change
				network.LastChangeHidden[i][j] = change
			end
		end
	end

	network.CalcError = function(target: {number}): number
		local errSum: number = 0.0
		for i = 1, #network.OutputLayer do
			local err: number = network.OutputLayer[i] - target[i]
			errSum += 0.5 * err * err
		end
		return errSum
	end

	network.Forward = function(input: {number}): {number}
		if #input ~= #network.InputLayer then
			error("amount of input variable doesn't match")
		end
		for i = 1, #input do
			network.InputLayer[i] = input[i]
		end
		network.InputLayer[#network.InputLayer] = 1.0

		for i = 1, #network.HiddenLayer-1 do
			local sum: number = 0.0
			for j = 1, #network.InputLayer do
				sum += network.InputLayer[j] * network.WeightHidden[i][j]
			end
			network.HiddenLayer[i] = sigmoid(sum)
		end

		network.HiddenLayer[#network.HiddenLayer] = 1.0

		for i = 1, #network.OutputLayer do
			local sum: number = 0.0
			for j = 1, #network.HiddenLayer do
				sum += network.HiddenLayer[j] * network.WeightOutput[i][j]
			end
			network.OutputLayer[i] = network.Regression and sum or sigmoid(sum)
		end

		return network.OutputLayer
	end

	network.TrainMap = function(inputs: {{[string]: number}}, targets: Matrix, iteration: number): ()
		if #targets ~= #network.OutputLayer then
			error("amount of output variable doesn't match")
		end

		local iter_flag: number = -1
		for i=1, iteration do
			local idx_ary: {number} = genRandomIdx(#inputs)
			local cur_err: number = 0.0
			for j=1, #inputs do
				network.ForwardMap(inputs[idx_ary[j]])
				network.FeedbackMap(targets[idx_ary[j]], inputs[idx_ary[j]])
				cur_err += network.CalcError(targets[idx_ary[j]])
				if j%1000 == 0 then
					iter_flag = iter_flag ~= i and i or iter_flag
					if debugging then print(`iteration {i}th / progress {string.format("%.2f", j*100 / #inputs)}`) end
				end
			end
			if ((iteration >= 10 and i%(iteration/10) == 0) or iteration < 10) and debugging then
				print(`iteration {i}th MSE: {string.format("%.5f", cur_err / #inputs)}`)
			end
		end
		if debugging then print("done.") end
	end
	network.ForwardMap = function(input: { [string]: number }): {number}
		for k: string, v: number in input do
			local key: number? = tonumber(k)
			if not key then continue end
			network.InputLayer[key] = v
		end
		network.InputLayer[#network.InputLayer] = 1.0

		for i=1, #network.HiddenLayer-1 do
			local sum: number = 0.0
			for j: string in input do
				local jNumber: number? = tonumber(j)
				if not jNumber then continue end
				sum += network.InputLayer[jNumber] * network[i][j]
			end
			network.HiddenLayer[i] = sigmoid(sum)
		end

		network.InputLayer[#network.InputLayer] = 1.0

		for i=1, #network.OutputLayer do
			local sum: number = 0.0
			for j=1, #network.HiddenLayer do
				sum += network.HiddenLayer[j] * network.WeightOutput[i][j]
			end
			network.OutputLayer[i] = network.Regression and sum or sigmoid(sum)
		end

		return network.OutputLayer
	end
	network.FeedbackMap = function(target: {number}, input: {[string]: number})
		for i=1, #network.OutputLayer do
			network.ErrOutput[i] = network.OutputLayer[i] - target[i]
		end

		for i=1, #network.HiddenLayer do
			local err: number = 0.0
			for j=1, #network.OutputLayer do
				local l: number = network.ErrOutput[j] * network.WeightOutput[j][i]
				err += network.Regression and l or l * dsigmoid(network.OutputLayer[j])
			end
			network.ErrHidden[i] = err
		end

		for i=1, #network.OutputLayer do
			for j=1, #network.HiddenLayer do
				local change: number = 0.0
				local delta: number = 0.0
				delta = network.Regression and network.ErrOutput[i] or network.ErrOutput[i] * dsigmoid(network.OutputLayer[i])
				change = network.Rate1*delta*network.HiddenLayer[j] + network.Rate2*network.LastChangeOutput[i][j]
				network.WeightOutput[i][j] -= change
				network.LastChangeOutput[i][j] = change
			end
		end

		for i=1, #network.HiddenLayer-1 do
			for j: string in input do
				local jNumber: number? = tonumber(j)
				if not jNumber then continue end
				local delta: number = network.ErrHidden[i] * dsigmoid(network.HiddenLayer[i])
				local change: number = network.Rate1 * delta * network.InputLayer[jNumber] + network.Rate2*network.LastChangeHidden[i][jNumber]
				network.WeightHidden[i][jNumber] -= change
				network.LastChangeHidden[i][jNumber] = change
			end
		end

	end
end

function nn.SaveNN(NeuralNetwork: NeuralNetwork): string
	return HttpService:JSONEncode(NeuralNetwork)
end
function nn.LoadNN(Model: string): NeuralNetwork
	local data: NeuralNetwork = HttpService:JSONDecode(Model)
	InitMethods(data)
	return data
end



function nn.new(inputCount: number, hiddenCount: number, outputCount: number, regression: boolean, debugging: boolean?): NeuralNetwork
	local network = {} :: NeuralNetwork
	-- default
	local rate1: number = 0.25
	local rate2: number = 0.1
	local regression: boolean = true
	
	network.Regression = regression
	network.Rate1 = rate1
	network.Rate2 = rate2
	network.InputLayer = makeArray(inputCount)
	network.HiddenLayer = makeArray(hiddenCount)
	network.OutputLayer = makeArray(outputCount)
	
	network.ErrOutput = makeArray(outputCount)
	network.ErrHidden = makeArray(hiddenCount)
	
	network.WeightHidden = randomMatrix(hiddenCount, inputCount, -1.0, 1.0)
	network.WeightOutput = randomMatrix(outputCount, hiddenCount, -1.0, 1.0)
	
	network.LastChangeHidden = makeMatrix(hiddenCount, inputCount, 0.0)
	network.LastChangeOutput = makeMatrix(outputCount, hiddenCount, 0.0)
	
	InitMethods(network)
	
	return network
end


return nn
