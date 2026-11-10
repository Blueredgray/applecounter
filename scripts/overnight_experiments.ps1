# Последовательный ночной цикл полных прогонов.
# Один шаг = один дочерний Python-процесс, чтобы не превысить 8 ГБ RAM и ~7 ГБ VRAM.
# Перезапуск без ручной очистки безопасен: run_matrix.py возобновляет только валидные задачи.

$ErrorActionPreference = 'Continue'
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

$env:PYTHONPATH = Join-Path $Root 'src'
$env:PYTHONIOENCODING = 'utf-8'
if ($IsMacOS -or $IsLinux) {
    $Python = Join-Path $Root '.venv/bin/python'
} else {
    $Python = Join-Path $Root '.venv\Scripts\python.exe'
}
$OuterLog = Join-Path $Root 'outputs\overnight_experiments.log'

function Write-OuterLog([string] $Message) {
    $Line = '{0} | {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Add-Content -Path $OuterLog -Value $Line -Encoding UTF8
}

function Invoke-ExperimentStep([string] $Name, [string[]] $Arguments) {
    Write-OuterLog "START | $Name"
    & $Python @Arguments
    $Code = $LASTEXITCODE
    if ($Code -eq 0) {
        Write-OuterLog "DONE  | $Name | exit=0"
    } else {
        Write-OuterLog "FAIL  | $Name | exit=$Code"
    }
    return $Code
}

$Steps = @(
    @{
        Name = 'T1 HSV: BoT-SORT + DeepSORT'
        Arguments = @('scripts\run_matrix.py', '--dataset', 't1', '--only', '/hsv/')
    },
    @{
        Name = 'T1 YOLOv8n@640: BoT-SORT + DeepSORT'
        Arguments = @('scripts\run_matrix.py', '--dataset', 't1', '--only', '/yolo8n/')
    },
    @{
        Name = 'T1 YOLO26s@1280: BoT-SORT + DeepSORT'
        Arguments = @('scripts\run_matrix.py', '--dataset', 't1', '--only', '/yolo26s/')
    },
    @{
        Name = 'MinneApple val detection: HSV'
        Arguments = @(
            '-m', 'applecounter', 'bench',
            '--data', 'data\processed\minneapple',
            '--detector', 'hsv',
            '--out', 'outputs\minneapple_detection\bench_hsv.json'
        )
    },
    @{
        Name = 'MinneApple val detection: YOLOv8n@640'
        Arguments = @(
            '-m', 'applecounter', 'bench',
            '--data', 'data\processed\minneapple',
            '--detector', 'yolo',
            '--yolo-weights', 'weights\runs\minneapple50\weights\best.pt',
            '--set', 'detector.yolo.imgsz=640',
            '--out', 'outputs\minneapple_detection\bench_yolo8n_640.json'
        )
    },
    @{
        Name = 'MinneApple val detection: YOLO26s@1280'
        Arguments = @(
            '-m', 'applecounter', 'bench',
            '--data', 'data\processed\minneapple',
            '--detector', 'yolo',
            '--yolo-weights', 'weights\runs\minneapple26s1280\weights\best.pt',
            '--set', 'detector.yolo.imgsz=1280',
            '--out', 'outputs\minneapple_detection\bench_yolo26s_1280.json'
        )
    },
    @{
        Name = 'VFC HSV: 16 videos x BoT-SORT/DeepSORT'
        Arguments = @('scripts\run_matrix.py', '--dataset', 'vfc', '--only', '/hsv/')
    },
    @{
        Name = 'VFC YOLOv8n@640: 16 videos x BoT-SORT/DeepSORT'
        Arguments = @('scripts\run_matrix.py', '--dataset', 'vfc', '--only', '/yolo8n/')
    },
    @{
        Name = 'VFC YOLO26s@1280: 16 videos x BoT-SORT/DeepSORT'
        Arguments = @('scripts\run_matrix.py', '--dataset', 'vfc', '--only', '/yolo26s/')
    }
)

Write-OuterLog 'NIGHT RUN START'
$Failed = New-Object System.Collections.Generic.List[string]
foreach ($Step in $Steps) {
    $Code = Invoke-ExperimentStep -Name $Step.Name -Arguments $Step.Arguments
    if ($Code -ne 0) {
        $Failed.Add($Step.Name)
    }
}

if ($Failed.Count -eq 0) {
    Write-OuterLog 'NIGHT RUN COMPLETE | failed=0'
    exit 0
}

foreach ($Name in $Failed) {
    Write-OuterLog "NIGHT RUN COMPLETE WITH FAILURES | $Name"
}
exit 1
