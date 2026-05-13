# Fetch BTC kline (candle) data from Bybit v5 market API.
# Endpoint docs: https://bybit-exchange.github.io/docs/v5/market/kline
# Public endpoint - no API key required.

[CmdletBinding()]
param(
    [string]$Symbol   = 'BTCUSDT',
    [string]$Category = 'spot',     # spot | linear | inverse
    [int]   $Limit    = 200         # 1..1000
)

$ErrorActionPreference = 'Stop'
$BaseUrl = 'https://api.bybit.com/v5/market/kline'

# Bybit interval codes: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M.
$Timeframes = [ordered]@{
    '1m'  = '1'
    '15m' = '15'
    '1h'  = '60'
    '1d'  = 'D'
}

function Get-BybitKlines {
    param(
        [Parameter(Mandatory)] [string] $Symbol,
        [Parameter(Mandatory)] [string] $Category,
        [Parameter(Mandatory)] [string] $Interval,
        [Parameter(Mandatory)] [int]    $Limit
    )

    $query = "category=$Category&symbol=$Symbol&interval=$Interval&limit=$Limit"
    $url   = "${BaseUrl}?$query"

    $resp = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 15

    if ($resp.retCode -ne 0) {
        throw "Bybit API error retCode=$($resp.retCode) retMsg=$($resp.retMsg)"
    }

    # Bybit returns rows newest-first. Reverse so [0] is the oldest, [-1] is the most recent.
    $rows = @($resp.result.list)
    [array]::Reverse($rows)

    foreach ($r in $rows) {
        [pscustomobject]@{
            OpenTime  = [DateTimeOffset]::FromUnixTimeMilliseconds([int64]$r[0]).UtcDateTime
            Open      = [decimal]$r[1]
            High      = [decimal]$r[2]
            Low       = [decimal]$r[3]
            Close     = [decimal]$r[4]
            Volume    = [decimal]$r[5]
            Turnover  = [decimal]$r[6]
        }
    }
}

$results = [ordered]@{}

foreach ($tf in $Timeframes.GetEnumerator()) {
    Write-Host "Fetching $Symbol $($tf.Key) (interval=$($tf.Value)) ..." -ForegroundColor Cyan
    try {
        $candles = Get-BybitKlines -Symbol $Symbol -Category $Category -Interval $tf.Value -Limit $Limit
        $results[$tf.Key] = $candles
        Write-Host ("  OK - {0} candles, latest close = {1} @ {2} UTC" -f `
            $candles.Count, $candles[-1].Close, $candles[-1].OpenTime.ToString('yyyy-MM-dd HH:mm:ss')) `
            -ForegroundColor Green
    }
    catch {
        Write-Host "  FAILED: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Latest candle per timeframe:" -ForegroundColor Yellow
foreach ($tf in $results.Keys) {
    $latest = $results[$tf][-1]
    "{0,-4}  open={1,-10}  high={2,-10}  low={3,-10}  close={4,-10}  vol={5}" -f `
        $tf, $latest.Open, $latest.High, $latest.Low, $latest.Close, $latest.Volume
}

# Expose for the caller (dot-source the script to access $BybitKlines in your session)
$Global:BybitKlines = $results
