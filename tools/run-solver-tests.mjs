import {spawn} from 'node:child_process';
import {chromium} from 'playwright';

const port=5189;
const server=spawn(process.execPath,['node_modules/vite/bin/vite.js','--host','127.0.0.1','--port',String(port),'--strictPort'],{stdio:['ignore','ignore','pipe']});
let browser;
try {
    let ready=false;
    for(let attempt=0;attempt<80;attempt++) {
        if(server.exitCode!==null)throw Error('Test server exited; port 5189 may be in use');
        try {ready=(await fetch(`http://127.0.0.1:${port}/`)).ok;} catch {}
        if(ready)break;
        await new Promise(resolve=>setTimeout(resolve,250));
    }
    if(!ready)throw Error('Test server did not start');
    const args=['--enable-unsafe-webgpu','--force-high-performance-gpu'];
    if(process.env.TEST_SOFTWARE_GPU==='1')args.push('--use-webgpu-adapter=swiftshader','--use-angle=swiftshader','--enable-unsafe-swiftshader');
    browser=await chromium.launch({channel:process.env.TEST_BROWSER_CHANNEL||'chromium',headless:true,args});
    for(const precision of ['32','16']) {
        const page=await browser.newPage();
        page.on('console',message=>{if(message.type()==='error')console.error(message.text());else if(message.type()==='log')console.log(message.text());});
        page.on('pageerror',error=>console.error(error.message));
        const url=`http://127.0.0.1:${port}/tests/solver.html?res=low&domain=local&checkDivergence&precision=${precision}&state=${precision}`;
        await page.goto(url,{waitUntil:'domcontentloaded',timeout:120000});
        await page.waitForFunction(()=>window.solverTests!==undefined);
        let timeout;
        try {
            const results=await Promise.race([page.evaluate(()=>window.solverTests),new Promise((_,reject)=>{
                timeout=setTimeout(()=>reject(Error('GPU suite exceeded five minutes')),300000);
            })]);
            console.log(`${precision}-bit: ${results.length} checks passed`);
            for(const result of results)console.log(`  PASS ${result.name}`);
        }finally{clearTimeout(timeout);await page.close();}
    }
}catch(error){console.error(error);process.exitCode=1;}
finally{await browser?.close();server.kill();}
