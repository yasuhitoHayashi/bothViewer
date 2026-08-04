const API='http://127.0.0.1:5000';
const $=id=>document.getElementById(id);
async function loadCameras(){
  $('cameraMessage').textContent='カメラを検出しています…';
  try{
    const response=await fetch(`${API}/api/cameras`);const data=await response.json();
    const cameras=data.cameras||[];
    for(const id of ['evs1','evs2']){
      const first=$(id).options[0];$(id).replaceChildren(first);
      cameras.forEach(serial=>{const option=document.createElement('option');option.value=serial;option.textContent=serial;$(id).append(option);});
    }
    if(cameras[0])$('evs1').value=cameras[0];if(cameras[1])$('evs2').value=cameras[1];
    $('cameraMessage').textContent=cameras.length?`${cameras.length}台のEVSを検出しました。`:'EVSは検出されませんでした。保存データの確認は利用できます。';
  }catch(error){$('cameraMessage').textContent=`検出できません: ${error.message}`;}
}
async function launch(mode){
  document.querySelectorAll('.mode-card').forEach(button=>button.disabled=true);
  $('launchMessage').className='message';$('launchMessage').textContent='サービスを準備しています…';
  try{
    const response=await fetch(`${API}/api/launch`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode,evs1:$('evs1').value,evs2:$('evs2').value})});
    const data=await response.json();if(!response.ok)throw new Error(data.message||`HTTP ${response.status}`);
    window.location.href=data.viewer_url;
  }catch(error){$('launchMessage').className='message error';$('launchMessage').textContent=error.message;document.querySelectorAll('.mode-card').forEach(button=>button.disabled=false);}
}
document.querySelectorAll('.mode-card').forEach(button=>button.addEventListener('click',()=>launch(button.dataset.mode)));
$('refreshCameras').addEventListener('click',loadCameras);loadCameras();
