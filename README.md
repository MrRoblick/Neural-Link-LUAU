# Easy to use


Example of usage
```lua
local ReplicatedStorage = game:GetService('ReplicatedStorage')
local NeuralLink = require(ReplicatedStorage.NeuralLink)
local neural = NeuralLink.new(3, 16, 4, true)
local input: NeuralLink.Matrix = {
	{0.5, 1, 1}, {0.9, 1, 2}, {0.8, 0, 1},
	{0.3, 1, 1}, {0.6, 1, 2}, {0.4, 0, 1},
	{0.9, 1, 7}, {0.6, 1, 4}, {0.1, 0, 1},
	{0.6, 1, 0}, {1, 0, 0},
}
local target: NeuralLink.Matrix = {
	{1,0,0,0}, {1,0,0,0}, {1,0,0,0},
	{0,1,0,0}, {0,1,0,0}, {0,1,0,0},
	{0,0,1,0}, {0,0,1,0}, {0,0,1,0},
	{0,0,0,1}, {0,0,0,1},
}
neural.Train(input, target, 100, false)

local function Action(output: {number}): string
	local max: number = -99999
	local pos: number = -1
	local actions: {string} = {
		"Атаковать",
		"Красться",
		"Убегать",
		"Ничего не делать",
	}
	
	for i: number, v: number in ipairs(output) do
		if v > max then
			max = v
			pos = i
		end
	end
	
	return actions[pos]
	
end
local hp: number = 0.1
local weapon: number = 0.0
local enemyCount: number = 15.0

print(Action(neural.Forward({hp, weapon, enemyCount})))
```
