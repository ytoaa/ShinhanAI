:: change root path
set root=C:\Users\wff\miniconda3
call %root%\Scripts\activate.bat %root%

call conda env list
call conda activate ai
:: change python project path
call cd C:\Users\wff\Documents\Shinhan\ShinhanAI
jupyter lab