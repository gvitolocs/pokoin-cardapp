const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const html = fs.readFileSync(process.argv[2] || path.join(__dirname, 'index.html'), 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function scanner() {
  const draws = [], navigations = [], requests = [], timers = [];
  const ctx = {clearRect(){}, strokeRect(){}, drawImage(...args){draws.push(args);}};
  const elements = new Map();
  function node(id='') {
    const classes=new Set();
    return { hidden:id==='shot', clientWidth:390,clientHeight:680,videoWidth:1280,videoHeight:720,
      attributes:{},children:[],textContent:'',
      classList:{add:k=>classes.add(k),remove:k=>classes.delete(k),contains:k=>classes.has(k),toggle(k,on){if(on)classes.add(k);else classes.delete(k)}},
      setAttribute(k,v){this.attributes[k]=v},getAttribute(k){return this.attributes[k]},removeAttribute(k){delete this.attributes[k]},
      append(...items){this.children.push(...items)},replaceChildren(...items){this.children=items},addEventListener(){},
      getContext(){return ctx},toBlob(callback){callback({width:this.width,height:this.height})},
    };
  }
  function element(id) {
    if (!elements.has(id)) elements.set(id,node(id));
    return elements.get(id);
  }
  const track = {stop(){},getSettings:()=>({deviceId:"rear"}),applyConstraints: async constraints => requests.push(constraints)};
  const sandbox = {
    document: {getElementById:element, createElement:()=>node()},
    navigator: {mediaDevices:{getUserMedia: async constraints => {
      requests.push(constraints.video); return {getVideoTracks:()=>[track],getTracks:()=>[track]};
    }}},
    window: {addEventListener(){}}, screen:{}, devicePixelRatio:1,
    location:{hostname:'scan.pokoin.com',href:'https://scan.pokoin.com/',assign:url=>navigations.push(url)},
    performance:{now:()=>0}, requestAnimationFrame:()=>1,
    setTimeout:callback=>{timers.push(callback); return timers.length;}, clearTimeout(){},
    AbortController, AbortSignal,
    cancelAnimationFrame(){},
    history:{replaceState(_a,_b,url){sandbox.location.href=String(url)}},
    fetch:async()=>({ok:true,json:async()=>({catalogs:['pokemon_western','pokemon_japanese','pokemon_chinese','pokemon_generic','one_piece_english','one_piece_japanese','one_piece_singles','riftbound_western'].map(id=>({id}))})}),
    URL, FormData:class {append(){}},
  };
  const context = vm.createContext(sandbox);
  vm.runInContext(source,context);
  return {run:code=>vm.runInContext(code,context),sandbox,elements,element,draws,navigations,requests,timers};
}

test('portrait preview and capture keep the entire sensor frame without software zoom',async()=>{
 const s=scanner();
 assert.match(html,/#live \{ object-fit: contain;/);
 const blob=await s.run('blobFromCanvas(live,960)');
 assert.deepEqual(blob,{width:960,height:540});
 assert.deepEqual(s.draws[0].slice(1),[0,0,1280,720,0,0,960,540]);
 const box=s.run('mapContain([0,0,960,540],960,540,390,680)');
 assert.equal(box.x,0);assert.equal(box.w,390);assert.equal(box.h,219.375);assert.equal(box.y,230.3125);
});
test('sensor constraints stay in primary orientation during portrait, landscape and browser chrome resizing',async()=>{
 const s=scanner();await turn();const initial=JSON.stringify(s.run('cameraConstraints()'));
 assert.equal(s.run('cameraConstraints().width.ideal'),1280);
 assert.equal(s.run('cameraConstraints().height.ideal'),960);
 assert.equal(s.run('cameraConstraints().aspectRatio.ideal'),4/3);
 assert.equal(s.run('cameraConstraints().resizeMode.ideal'),'none');
 const before=s.requests.length;
 for(const [w,h] of [[680,390],[390,500],[390,664]]){
  Object.assign(s.element('overlay'),{clientWidth:w,clientHeight:h});
  s.run('cameraViewChanged()');
  assert.equal(JSON.stringify(s.run('cameraConstraints()')),initial);
 }
 assert.equal(s.requests.length,before);
});
test('portrait native frame retains all pixels and aligns the detected box without rotation or zoom',async()=>{
 const s=scanner();Object.assign(s.element('live'),{videoWidth:960,videoHeight:1280});
 assert.deepEqual(await s.run('blobFromCanvas(live,960)'),{width:720,height:960});
 assert.deepEqual(s.draws[0].slice(1),[0,0,960,1280,0,0,720,960]);
 const box=s.run('mapContain([180,240,540,720],720,960,390,664)');
 assert.equal(box.x,97.5);assert.equal(box.y,202);assert.equal(box.w,195);assert.equal(box.h,260);
});

test('pokemon lock overlay snaps YOLO box to 63:88 card aspect',()=>{
 const s=scanner();
 const tall=s.run('fitPokemonCardRect({x:100,y:100,w:100,h:300})');
 assert.ok(Math.abs(tall.w / tall.h - 63/88) < 1e-6);
 assert.equal(tall.h, 300);
 assert.ok(Math.abs(tall.w - 300 * 63/88) < 1e-6);
 const wide=s.run('fitPokemonCardRect({x:0,y:0,w:200,h:100})');
 assert.ok(Math.abs(wide.w / wide.h - 63/88) < 1e-6);
 assert.equal(wide.w, 200);
});

test('gallery keeps the full image without applying the camera crop',async()=>{
  const s=scanner();
  const blob=await s.run('blobFromCanvas({naturalWidth:1600,naturalHeight:900},1280)');
  assert.deepEqual(blob,{width:1280,height:720});
  assert.deepEqual(s.draws[0].slice(1),[0,0,1600,900,0,0,1280,720]);
});

test('a delayed response from before rotation cannot draw boxes or navigate',async()=>{
  const s=scanner();
  await new Promise(resolve=>setImmediate(resolve));
  let reply;
  s.sandbox.fetch=()=>new Promise(resolve=>{reply=resolve;});
  const pending=s.run('identify({}, {live:true,viewVersion:cameraViewVersion})');
  s.run('invalidateCameraView()');
  reply({ok:true,json:async()=>({immediate:true,img_w:960,img_h:540,
    boxes:[{xyxy:[100,100,200,300]}],hits:[{pokoin_url:'https://pokoin.com/card/123'}]})});
  assert.equal(await pending,null);
  assert.equal(s.run('lastBoxes'),null);
  assert.deepEqual(s.navigations,[]);
});

const turn=()=>new Promise(resolve=>setImmediate(resolve));
test('all six catalog choices route to their own API and game-specific language controls',async()=>{
 const s=scanner();await turn();
 for(const [game,language] of [['pokemon','western'],['pokemon','japanese'],['pokemon','chinese'],['one_piece','english'],['one_piece','japanese'],['riftbound','western']]){
  s.run(`selectCatalog(${JSON.stringify(game)},${JSON.stringify(language)})`);
  assert.equal(new URL(s.run('identifyEndpoint(true)')).searchParams.get('catalog'),game+'_'+language);
  assert.equal(s.element('langCn').hidden,game!=='pokemon');
  assert.equal(s.element('langJp').hidden,game==='riftbound');
 }
});
test('generic singles and one piece singles catalogs are extra leftover-JPEG galleries',async()=>{
 const s=scanner();await turn();
 s.run('selectCatalog("pokemon","generic")');
 assert.equal(new URL(s.run('identifyEndpoint(true)')).searchParams.get('catalog'),'pokemon_generic');
 assert.equal(s.element('langExtra').hidden,false);
 assert.equal(s.element('langExtra').textContent,'GS');
 s.run('selectCatalog("one_piece","singles")');
 assert.equal(new URL(s.run('identifyEndpoint(true)')).searchParams.get('catalog'),'one_piece_singles');
 assert.equal(s.element('langExtra').textContent,'SG');
 s.run('selectCatalog("riftbound","western")');
 assert.equal(s.element('langExtra').hidden,true);
});
test('switching catalog discards an in-flight match',async()=>{
 const s=scanner();await turn();let reply;
 s.sandbox.fetch=()=>new Promise(resolve=>reply=resolve);
 const pending=s.run('identify({}, {live:true,viewVersion:cameraViewVersion})');
 s.run('selectCatalog("one_piece","japanese")');
 reply({ok:true,json:async()=>({catalog:'pokemon_western',immediate:true,top1:{id:'219698',public_id:'219698',score:.99,pokoin_url:'https://pokoin.com/219698'},boxes:[{xyxy:[1,2,3,4]}]})});
 assert.equal(await pending,null);assert.equal(s.run('lastBoxes'),null);assert.deepEqual(s.navigations,[]);
});
test('single navigation uses the top match only and rejects the wrong game hostname',async()=>{
 const s=scanner();await turn();s.run('selectCatalog("one_piece","english")');
 assert.equal(s.run('pokoinUrl({top1:{public_id:"488352",pokoin_url:"https://onepiece.pokoin.com/488352"}})'), 'https://onepiece.pokoin.com/488352');
 assert.equal(s.run('pokoinUrl({top1:{public_id:"488352",pokoin_url:"https://pokoin.com/488352"}})'), '');
 assert.equal(s.run('pokoinUrl({top1:{id:"not-mapped"},hits:[{public_id:"488352",pokoin_url:"https://onepiece.pokoin.com/488352"}]})'), '');
});
test('multi shows distinct detected cards without auto-navigation',async()=>{
 const s=scanner();await turn();s.run('setMode("multi")');
 const top={id:'219698',public_id:'219698',score:.99,pokoin_url:'https://pokoin.com/219698',name:'Oddish'};
 s.sandbox.fetch=async()=>({ok:true,json:async()=>({catalog:'pokemon_western',immediate:true,top1:top,hits:[top],cards:[{top1:top},{top1:{...top,id:'243376',public_id:'243376',pokoin_url:'https://pokoin.com/243376',name:'Grimer'}}]})});
 await s.run('identify({}, {live:true,viewVersion:cameraViewVersion})');
 assert.equal(s.element('multiList').children.length,2);assert.deepEqual(s.navigations,[]);
});
test('gallery results arriving after a catalog switch are also ignored',async()=>{
 const s=scanner();await turn();let reply;s.sandbox.fetch=()=>new Promise(resolve=>reply=resolve);
 const pending=s.run('identify({}, {live:false})');s.run('selectCatalog("pokemon","chinese")');
 reply({ok:true,json:async()=>({catalog:'pokemon_western',top1:{id:'219698',public_id:'219698',score:.99,pokoin_url:'https://pokoin.com/219698'}})});
 assert.equal(await pending,null);assert.deepEqual(s.navigations,[]);
});

test('normal physical rear lens wins over telephoto, ultra-wide, virtual and front lenses',async()=>{
 const s=scanner();await turn();
 for(const normal of ['Back Camera','Back Wide Angle Camera','Bagsidekamera','Fotocamera posteriore grandangolare']){
  const labels=['Back Triple Camera','Back Telephoto Camera','Back Ultra Wide Camera','Front Camera',normal];
  const devices=labels.map((label,i)=>({kind:'videoinput',deviceId:String(i),label}));
  assert.equal(s.run(`preferredCamera(${JSON.stringify(devices)},"0").deviceId`),'4');
 }
 assert.equal(s.run('preferredCamera([{kind:"videoinput",deviceId:"a",label:""},{kind:"videoinput",deviceId:"b",label:""}],"b").deviceId'),'b');
});
test('camera startup reopens the normal lens, resets zoom and lets users select a different lens',async()=>{
 const s=scanner();await turn();const stopped=[],opened=[],applied=[];
 s.sandbox.navigator.mediaDevices={
  enumerateDevices:async()=>['Back Ultra Wide Camera','Back Wide Angle Camera','Back Telephoto Camera'].map((label,i)=>({kind:'videoinput',deviceId:String(i),label})),
  getUserMedia:async({video})=>{const id=video.deviceId?.exact||'0';opened.push(id);const track={stop:()=>stopped.push(id),getSettings:()=>({deviceId:id}),getCapabilities:()=>({zoom:{min:1,max:9},focusMode:['continuous']}),applyConstraints:async(c)=>applied.push(c)};return {getVideoTracks:()=>[track],getTracks:()=>[track]};}
 };
 await s.run('startCam()');assert.deepEqual(opened,['0','1']);assert.ok(stopped.includes('0'));
 assert.equal(s.element('cameraControls').hidden,false);assert.equal(s.element('cameraSelect').children.length,3);
 assert.equal(applied[0].width.ideal,1280);assert.equal(applied[0].height.ideal,960);assert.equal(applied[0].aspectRatio.ideal,4/3);assert.equal(applied[0].advanced[0].zoom,1);assert.equal(applied[0].advanced[0].focusMode,'continuous');
 await s.run('startCam("2")');assert.equal(opened.at(-1),'2');assert.equal(s.run('track.getSettings().deviceId'),'2');
});
test('late camera permission response is stopped after a newer camera request',async()=>{
 const s=scanner();await turn();let resolveOld;const stopped=[];
 const stream=id=>{const track={stop:()=>stopped.push(id),getSettings:()=>({deviceId:id})};return {getVideoTracks:()=>[track],getTracks:()=>[track]};};
 s.sandbox.navigator.mediaDevices={enumerateDevices:async()=>[],getUserMedia:async({video})=>video.deviceId.exact==='old'?new Promise(r=>resolveOld=r):stream('new')};
 const old=s.run('startCam("old")');await s.run('startCam("new")');resolveOld(stream('old'));await old;
 assert.deepEqual(stopped,['old']);assert.equal(s.run('track.getSettings().deviceId'),'new');
});
test('weak live match offers a clickable candidate without automatic navigation',async()=>{
 const s=scanner();await turn();s.run('selectCatalog("one_piece","english")');
 const top={id:'489018',public_id:'489018',score:.715,name:'Monkey.D.Luffy',pokoin_url:'https://onepiece.pokoin.com/489018'};
 s.sandbox.fetch=async()=>({ok:true,json:async()=>({catalog:'one_piece_english',top1:top,boxes:[]})});
 await s.run('identify({}, {live:true,viewVersion:cameraViewVersion})');
 assert.equal(s.element('multiList').children[0].href,top.pokoin_url);assert.match(s.element('warming').textContent,/Possible match/);assert.deepEqual(s.navigations,[]);
});
test('the actual accepted Luffy API response opens its card on the first frame',async()=>{
 const s=scanner();await turn();s.run('selectCatalog("one_piece","english")');
 const response=JSON.parse(fs.readFileSync(path.join(__dirname,'fixtures/scanner-redirect/luffy-response.json'),'utf8'));
 assert.equal(response.immediate,false);assert.equal(response.top1.score,.78);
 s.sandbox.fetch=async()=>({ok:true,json:async()=>response});
 await s.run('identify({}, {live:true,viewVersion:cameraViewVersion})');
 assert.deepEqual(s.navigations,[response.top1.pokoin_url]);
 assert.match(s.element('warming').textContent,/Opening/);
 assert.equal(s.element('multiList').children[0].href,response.top1.pokoin_url);
});

test('browser chrome resize during identification does not discard a valid full-sensor response',async()=>{
 const s=scanner();await turn();let reply;
 s.run('selectCatalog("one_piece","english")');
 const response=JSON.parse(fs.readFileSync(path.join(__dirname,'fixtures/scanner-redirect/luffy-response.json'),'utf8'));
 s.sandbox.fetch=()=>new Promise(resolve=>reply=resolve);
 const pending=s.run('identify({}, {live:true,viewVersion:cameraViewVersion})');
 s.element('overlay').clientHeight=620;s.run('cameraViewChanged()');
 reply({ok:true,json:async()=>response});
 assert.ok(await pending);assert.deepEqual(s.navigations,[response.top1.pokoin_url]);
});
test('accepted Japanese gallery response opens the exact public card',async()=>{
 const s=scanner();await turn();s.run('selectCatalog("one_piece","japanese")');
 const response=JSON.parse(fs.readFileSync(path.join(__dirname,'fixtures/scanner-redirect/shanks-response.json'),'utf8'));
 s.sandbox.fetch=async()=>({ok:true,json:async()=>response});
 await s.run('identify({}, {live:false})');
 assert.deepEqual(s.navigations,[response.top1.pokoin_url]);
});
test('a failed navigation keeps a usable card link and explains the next action',async()=>{
 const s=scanner();await turn();s.run('selectCatalog("one_piece","english")');
 const response=JSON.parse(fs.readFileSync(path.join(__dirname,'fixtures/scanner-redirect/luffy-response.json'),'utf8'));
 s.sandbox.fetch=async()=>({ok:true,json:async()=>response});
 s.sandbox.location.assign=()=>{throw new Error('navigation denied')};
 await s.run('identify({}, {live:true,viewVersion:cameraViewVersion})');
 assert.equal(s.element('multiList').children[0].href,response.top1.pokoin_url);
 assert.match(s.element('warming').textContent,/Tap.*open/);
});
