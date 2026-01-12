import React, { useState } from 'react';

const TaxonomyDiagram = () => {
  const [activeStep, setActiveStep] = useState(0);
  
  const steps = [
    { id: 0, title: 'Framework Overview', icon: '🏛️' },
    { id: 1, title: 'Level 1: Lexical Quality', icon: '✍️' },
    { id: 2, title: 'Level 2: Linguistic Quality', icon: '🌐' },
    { id: 3, title: 'Level 3: Islamic Context', icon: '🕌' },
    { id: 4, title: 'Temporal-Cultural Injection', icon: '📅' },
    { id: 5, title: 'Cross-Linguistic Unification', icon: '🔗' },
  ];

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0a1628 0%, #1a2744 50%, #0d1f3c 100%)',
      fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
      color: '#e8eef7',
      padding: '24px'
    }}>
      {/* Header */}
      <div style={{
        textAlign: 'center',
        marginBottom: '32px',
        padding: '24px',
        background: 'linear-gradient(90deg, rgba(34,139,134,0.2) 0%, rgba(59,130,246,0.2) 100%)',
        borderRadius: '16px',
        border: '1px solid rgba(34,139,134,0.3)'
      }}>
        <h1 style={{
          fontSize: '28px',
          fontWeight: '700',
          margin: '0 0 8px 0',
          background: 'linear-gradient(90deg, #22d3ee, #3b82f6, #22d3ee)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundSize: '200% auto',
        }}>
          AlHaram Analytics Framework
        </h1>
        <p style={{ 
          fontSize: '16px', 
          color: '#94a3b8', 
          margin: 0,
          letterSpacing: '0.5px'
        }}>
          Multilingual Data Quality Taxonomy for Islamic Service Analytics
        </p>
      </div>

      {/* Step Navigation */}
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        gap: '8px',
        marginBottom: '32px',
        flexWrap: 'wrap'
      }}>
        {steps.map((step) => (
          <button
            key={step.id}
            onClick={() => setActiveStep(step.id)}
            style={{
              padding: '12px 20px',
              borderRadius: '12px',
              border: activeStep === step.id 
                ? '2px solid #22d3ee' 
                : '2px solid rgba(148,163,184,0.2)',
              background: activeStep === step.id 
                ? 'linear-gradient(135deg, rgba(34,211,238,0.2), rgba(59,130,246,0.2))' 
                : 'rgba(30,41,59,0.5)',
              color: activeStep === step.id ? '#22d3ee' : '#94a3b8',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: '600',
              transition: 'all 0.3s ease',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            <span style={{ fontSize: '18px' }}>{step.icon}</span>
            <span>{step.title}</span>
          </button>
        ))}
      </div>

      {/* Content Area */}
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto'
      }}>
        {activeStep === 0 && <FrameworkOverview />}
        {activeStep === 1 && <LexicalQuality />}
        {activeStep === 2 && <LinguisticQuality />}
        {activeStep === 3 && <IslamicContext />}
        {activeStep === 4 && <TemporalCultural />}
        {activeStep === 5 && <CrossLinguistic />}
      </div>

      {/* Navigation Buttons */}
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        gap: '16px',
        marginTop: '32px'
      }}>
        <button
          onClick={() => setActiveStep(Math.max(0, activeStep - 1))}
          disabled={activeStep === 0}
          style={{
            padding: '12px 32px',
            borderRadius: '8px',
            border: 'none',
            background: activeStep === 0 ? 'rgba(148,163,184,0.2)' : 'rgba(34,211,238,0.2)',
            color: activeStep === 0 ? '#64748b' : '#22d3ee',
            cursor: activeStep === 0 ? 'not-allowed' : 'pointer',
            fontSize: '14px',
            fontWeight: '600'
          }}
        >
          ← Previous
        </button>
        <button
          onClick={() => setActiveStep(Math.min(5, activeStep + 1))}
          disabled={activeStep === 5}
          style={{
            padding: '12px 32px',
            borderRadius: '8px',
            border: 'none',
            background: activeStep === 5 ? 'rgba(148,163,184,0.2)' : 'linear-gradient(90deg, #22d3ee, #3b82f6)',
            color: activeStep === 5 ? '#64748b' : '#ffffff',
            cursor: activeStep === 5 ? 'not-allowed' : 'pointer',
            fontSize: '14px',
            fontWeight: '600'
          }}
        >
          Next →
        </button>
      </div>
    </div>
  );
};

// Step 0: Framework Overview
const FrameworkOverview = () => (
  <div>
    <SectionTitle>High-Level Framework Architecture</SectionTitle>
    
    {/* Main Flow Diagram */}
    <div style={{
      background: 'rgba(30,41,59,0.6)',
      borderRadius: '16px',
      padding: '32px',
      border: '1px solid rgba(148,163,184,0.1)'
    }}>
      {/* Input */}
      <FlowBox color="#f59e0b" title="INPUT: Multilingual App Reviews">
        <div style={{ fontSize: '13px', color: '#cbd5e1' }}>
          Reviews from 180+ countries in Arabic, Urdu, Indonesian, Turkish, English, Bengali, Malay, French, Hausa, Swahili...
        </div>
      </FlowBox>
      
      <FlowArrow />
      
      {/* Taxonomy Levels */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(34,211,238,0.1), rgba(59,130,246,0.1))',
        borderRadius: '12px',
        padding: '24px',
        border: '1px solid rgba(34,211,238,0.3)',
        marginBottom: '16px'
      }}>
        <div style={{ 
          fontSize: '14px', 
          fontWeight: '700', 
          color: '#22d3ee',
          marginBottom: '16px',
          textAlign: 'center'
        }}>
          DATA QUALITY TAXONOMY (4 Levels)
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
          <TaxonomyLevel 
            level="1" 
            title="Lexical Quality" 
            items={['Script Normalization', 'Identity Normalization', 'Islamic Terminology']}
            color="#f472b6"
          />
          <TaxonomyLevel 
            level="2" 
            title="Linguistic Quality" 
            items={['Language Detection', 'Script Family', 'Code-Mixing']}
            color="#a78bfa"
          />
          <TaxonomyLevel 
            level="3" 
            title="Islamic Context" 
            items={['Hijri Calendar', 'Pilgrimage Stage', 'Service Domain']}
            color="#34d399"
          />
          <TaxonomyLevel 
            level="4" 
            title="Demographic" 
            items={['Gender Prediction', 'Origin Inference', '(Optional)']}
            color="#fbbf24"
          />
        </div>
      </div>
      
      <FlowArrow />
      
      {/* Output */}
      <FlowBox color="#22c55e" title="OUTPUT: Enriched, Standardized Dataset">
        <div style={{ fontSize: '13px', color: '#cbd5e1' }}>
          Ready for: Sentiment Analysis • Cross-Linguistic Studies • Service Quality Research • User Behavior Modeling
        </div>
      </FlowBox>
    </div>
    
    {/* Key Innovation Boxes */}
    <div style={{ 
      display: 'grid', 
      gridTemplateColumns: '1fr 1fr', 
      gap: '16px',
      marginTop: '24px'
    }}>
      <InnovationBox 
        icon="📊"
        title="Innovation 1: Data Quality Taxonomy"
        description="Formalizes preprocessing as systematic, reusable quality dimensions specific to multilingual Islamic service analytics"
      />
      <InnovationBox 
        icon="🔄"
        title="Innovation 2: Cross-Linguistic Unification"
        description="Islamic temporal-cultural context serves as shared semantic layer across all languages"
      />
    </div>
  </div>
);

// Step 1: Lexical Quality
const LexicalQuality = () => (
  <div>
    <SectionTitle>Level 1: Cross-Linguistic Lexical Quality</SectionTitle>
    
    {/* 1.1 Script Normalization */}
    <SubSection title="1.1 Script Normalization" color="#f472b6">
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <ExampleCard 
          title="Arabizi Resolution"
          subtitle="Arabic written in Latin + numerals"
          examples={[
            { before: "a7mad", after: "ahmad", note: "7 → ح (Ha)" },
            { before: "5aled", after: "khaled", note: "5 → خ (Kha)" },
            { before: "3ali", after: "ali", note: "3 → ع (Ain)" },
            { before: "mara7ba", after: "marahba", note: "7 → ح" }
          ]}
        />
        <ExampleCard 
          title="Roman Urdu"
          subtitle="Urdu written in Latin script"
          examples={[
            { before: "Shukriya", after: "شکریہ", note: "Thank you" },
            { before: "JazakAllah", after: "جزاک اللہ", note: "May Allah reward" },
            { before: "bohat acha", after: "بہت اچھا", note: "Very good" }
          ]}
        />
      </div>
      
      <div style={{ 
        marginTop: '16px',
        padding: '16px',
        background: 'rgba(244,114,182,0.1)',
        borderRadius: '8px',
        border: '1px solid rgba(244,114,182,0.3)'
      }}>
        <div style={{ fontSize: '13px', fontWeight: '600', color: '#f472b6', marginBottom: '8px' }}>
          📝 Arabizi Character Mapping Table
        </div>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(5, 1fr)', 
          gap: '8px',
          fontSize: '12px'
        }}>
          {[
            { num: '2', ar: 'ء/أ', sound: 'Hamza' },
            { num: '3', ar: 'ع', sound: 'Ain' },
            { num: '5', ar: 'خ', sound: 'Kha' },
            { num: '6', ar: 'ط', sound: 'Ta' },
            { num: '7', ar: 'ح', sound: 'Ha' },
            { num: '8', ar: 'غ', sound: 'Ghain' },
            { num: '9', ar: 'ص', sound: 'Sad' }
          ].map((item, i) => (
            <div key={i} style={{
              background: 'rgba(30,41,59,0.8)',
              padding: '8px',
              borderRadius: '6px',
              textAlign: 'center'
            }}>
              <span style={{ color: '#f472b6', fontWeight: '700' }}>{item.num}</span>
              <span style={{ color: '#64748b' }}> → </span>
              <span style={{ color: '#e2e8f0' }}>{item.ar}</span>
              <div style={{ color: '#94a3b8', fontSize: '10px' }}>{item.sound}</div>
            </div>
          ))}
        </div>
      </div>
    </SubSection>
    
    {/* 1.2 Identity Normalization */}
    <SubSection title="1.2 Identity Normalization" color="#f472b6">
      <ExampleCard 
        title="Username Cleaning (Multilingual)"
        subtitle="Handles diverse naming patterns across cultures"
        examples={[
          { before: "Mo7amed_2020 🇸🇦", after: "Mohamed", note: "Arabic: Arabizi + year + emoji" },
          { before: "fatima.ahmed.123", after: "fatima ahmed", note: "Arabic: dots + numbers" },
          { before: "புதியவர்", after: "புதியவர்", note: "Tamil: preserved" },
          { before: "Budi_Santoso99", after: "Budi Santoso", note: "Indonesian: underscore + year" },
          { before: "AB", after: "Anonymous", note: "Too short (< 3 chars)" }
        ]}
      />
    </SubSection>
    
    {/* 1.3 Islamic Terminology */}
    <SubSection title="1.3 Islamic Terminology Normalization" color="#f472b6">
      <div style={{ 
        background: 'rgba(30,41,59,0.6)',
        borderRadius: '12px',
        padding: '20px',
        border: '1px solid rgba(148,163,184,0.1)'
      }}>
        <div style={{ fontSize: '13px', fontWeight: '600', color: '#f472b6', marginBottom: '16px' }}>
          🕋 Cross-Language Term Unification
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(148,163,184,0.2)' }}>
                <th style={{ padding: '8px', textAlign: 'left', color: '#94a3b8' }}>Canonical</th>
                <th style={{ padding: '8px', textAlign: 'left', color: '#94a3b8' }}>Arabic</th>
                <th style={{ padding: '8px', textAlign: 'left', color: '#94a3b8' }}>English</th>
                <th style={{ padding: '8px', textAlign: 'left', color: '#94a3b8' }}>Indonesian</th>
                <th style={{ padding: '8px', textAlign: 'left', color: '#94a3b8' }}>Turkish</th>
                <th style={{ padding: '8px', textAlign: 'left', color: '#94a3b8' }}>French</th>
              </tr>
            </thead>
            <tbody>
              {[
                { canon: 'HAJJ', ar: 'حج / الحج', en: 'Hajj, Hadj, Haj', id: 'Haji', tr: 'Hac', fr: 'Hadj' },
                { canon: 'UMRAH', ar: 'عمرة', en: 'Umrah, Omra', id: 'Umroh', tr: 'Umre', fr: 'Omra' },
                { canon: 'MAKKAH', ar: 'مكة المكرمة', en: 'Makkah, Mecca', id: 'Mekkah', tr: 'Mekke', fr: 'La Mecque' },
                { canon: 'TAWAF', ar: 'طواف', en: 'Tawaf', id: 'Tawaf', tr: 'Tavaf', fr: 'Tawaf' }
              ].map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid rgba(148,163,184,0.1)' }}>
                  <td style={{ padding: '8px', color: '#22d3ee', fontWeight: '600' }}>{row.canon}</td>
                  <td style={{ padding: '8px', color: '#e2e8f0' }}>{row.ar}</td>
                  <td style={{ padding: '8px', color: '#e2e8f0' }}>{row.en}</td>
                  <td style={{ padding: '8px', color: '#e2e8f0' }}>{row.id}</td>
                  <td style={{ padding: '8px', color: '#e2e8f0' }}>{row.tr}</td>
                  <td style={{ padding: '8px', color: '#e2e8f0' }}>{row.fr}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </SubSection>
  </div>
);

// Step 2: Linguistic Quality
const LinguisticQuality = () => (
  <div>
    <SectionTitle>Level 2: Multilingual Linguistic Quality</SectionTitle>
    
    {/* 2.1 Language Detection */}
    <SubSection title="2.1 Language Identification" color="#a78bfa">
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', marginBottom: '16px' }}>
        <LanguageTierCard 
          tier="High-Resource"
          languages={['Arabic', 'English', 'Indonesian', 'Turkish']}
          accuracy="95%+"
          color="#22c55e"
        />
        <LanguageTierCard 
          tier="Medium-Resource"
          languages={['Urdu', 'Bengali', 'Malay', 'Persian', 'French']}
          accuracy="85-95%"
          color="#f59e0b"
        />
        <LanguageTierCard 
          tier="Lower-Resource"
          languages={['Hausa', 'Swahili', 'Pashto', 'Tamil', 'Somali']}
          accuracy="70-85%"
          color="#ef4444"
        />
      </div>
      
      <ExampleCard 
        title="Language Detection Examples"
        subtitle="Multi-strategy detection algorithm"
        examples={[
          { before: "التطبيق ممتاز جداً", after: "Arabic", note: "Arabic Unicode (U+0600-U+06FF)" },
          { before: "Aplikasi ini sangat bagus", after: "Indonesian", note: "Latin script + langid" },
          { before: "The app is very good التطبيق", after: "Mixed", note: "Both scripts present" },
          { before: "👍👍👍🔥", after: "Unknown", note: "Emojis only" }
        ]}
      />
    </SubSection>
    
    {/* 2.2 Script Family */}
    <SubSection title="2.2 Script Family Detection" color="#a78bfa">
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '12px'
      }}>
        <ScriptFamilyCard 
          family="Arabic Script Family"
          languages={['Arabic', 'Urdu', 'Persian', 'Pashto']}
          example="العربية • اردو • فارسی"
          color="#22d3ee"
        />
        <ScriptFamilyCard 
          family="Latin Script Family"
          languages={['English', 'Indonesian', 'Turkish', 'Malay', 'French', 'Hausa', 'Swahili']}
          example="English • Bahasa • Türkçe"
          color="#a78bfa"
        />
        <ScriptFamilyCard 
          family="Brahmic Script Family"
          languages={['Bengali', 'Tamil', 'Hindi']}
          example="বাংলা • தமிழ் • हिंदी"
          color="#f472b6"
        />
      </div>
    </SubSection>
    
    {/* 2.3 Code-Mixing */}
    <SubSection title="2.3 Code-Mixing Patterns" color="#a78bfa">
      <div style={{
        background: 'rgba(30,41,59,0.6)',
        borderRadius: '12px',
        padding: '20px',
        border: '1px solid rgba(148,163,184,0.1)'
      }}>
        <div style={{ fontSize: '13px', fontWeight: '600', color: '#a78bfa', marginBottom: '16px' }}>
          🔀 Common Code-Mixing Patterns in Islamic Service Reviews
        </div>
        <div style={{ display: 'grid', gap: '12px' }}>
          {[
            { 
              pattern: "Native + Islamic Terms", 
              example: "Aplikasi bagus untuk Hajj dan Umrah", 
              langs: "Indonesian + Arabic loanwords",
              note: "Universal pattern across all languages"
            },
            { 
              pattern: "Native + English Technical", 
              example: "App crash ho gaya registration ke waqt", 
              langs: "Urdu + English",
              note: "Technical terms often in English"
            },
            { 
              pattern: "Triple Mixing", 
              example: "Tawaf ke time app working نہیں تھی", 
              langs: "Arabic + English + Urdu",
              note: "Complex multilingual expression"
            }
          ].map((item, i) => (
            <div key={i} style={{
              background: 'rgba(167,139,250,0.1)',
              padding: '12px',
              borderRadius: '8px',
              border: '1px solid rgba(167,139,250,0.2)'
            }}>
              <div style={{ fontSize: '12px', color: '#a78bfa', fontWeight: '600' }}>{item.pattern}</div>
              <div style={{ fontSize: '14px', color: '#e2e8f0', margin: '4px 0', fontStyle: 'italic' }}>"{item.example}"</div>
              <div style={{ fontSize: '11px', color: '#94a3b8' }}>{item.langs} — {item.note}</div>
            </div>
          ))}
        </div>
      </div>
    </SubSection>
  </div>
);

// Step 3: Islamic Context
const IslamicContext = () => (
  <div>
    <SectionTitle>Level 3: Islamic Contextual Quality</SectionTitle>
    <div style={{ 
      fontSize: '14px', 
      color: '#94a3b8', 
      textAlign: 'center',
      marginBottom: '24px',
      fontStyle: 'italic'
    }}>
      The unifying layer that connects all languages through shared religious context
    </div>
    
    {/* 3.1 Temporal-Religious */}
    <SubSection title="3.1 Temporal-Religious Contextualization" color="#34d399">
      <div style={{
        background: 'rgba(30,41,59,0.6)',
        borderRadius: '12px',
        padding: '20px',
        border: '1px solid rgba(148,163,184,0.1)',
        marginBottom: '16px'
      }}>
        <div style={{ fontSize: '13px', fontWeight: '600', color: '#34d399', marginBottom: '16px' }}>
          📅 Hijri Calendar Mapping (Gregorian → Hijri → Islamic Period)
        </div>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '16px',
          flexWrap: 'wrap'
        }}>
          <div style={{ 
            background: 'rgba(245,158,11,0.2)', 
            padding: '12px 20px', 
            borderRadius: '8px',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '11px', color: '#f59e0b' }}>INPUT</div>
            <div style={{ fontSize: '16px', color: '#e2e8f0', fontWeight: '600' }}>2024-06-15</div>
            <div style={{ fontSize: '11px', color: '#94a3b8' }}>Gregorian</div>
          </div>
          <div style={{ color: '#64748b', fontSize: '24px' }}>→</div>
          <div style={{ 
            background: 'rgba(34,211,238,0.2)', 
            padding: '12px 20px', 
            borderRadius: '8px',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '11px', color: '#22d3ee' }}>CONVERT</div>
            <div style={{ fontSize: '16px', color: '#e2e8f0', fontWeight: '600' }}>1446-12-08</div>
            <div style={{ fontSize: '11px', color: '#94a3b8' }}>Dhul Hijjah 8</div>
          </div>
          <div style={{ color: '#64748b', fontSize: '24px' }}>→</div>
          <div style={{ 
            background: 'rgba(52,211,153,0.2)', 
            padding: '12px 20px', 
            borderRadius: '8px',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '11px', color: '#34d399' }}>OUTPUT</div>
            <div style={{ fontSize: '16px', color: '#e2e8f0', fontWeight: '600' }}>Hajj Season</div>
            <div style={{ fontSize: '11px', color: '#94a3b8' }}>Peak Pilgrimage</div>
          </div>
        </div>
      </div>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
        {[
          { period: 'Hajj Season', hijri: 'Dhul Hijjah 1-15', icon: '🕋', color: '#22d3ee' },
          { period: 'Eid al-Adha', hijri: 'Dhul Hijjah 10-13', icon: '🎉', color: '#f472b6' },
          { period: 'Ramadan', hijri: 'Month 9 (full)', icon: '🌙', color: '#a78bfa' },
          { period: 'Eid al-Fitr', hijri: 'Shawwal 1-3', icon: '✨', color: '#34d399' },
          { period: 'Friday/Jumu\'ah', hijri: 'Every Friday', icon: '🕌', color: '#f59e0b' },
          { period: 'Regular', hijri: 'Other dates', icon: '📆', color: '#64748b' }
        ].map((item, i) => (
          <div key={i} style={{
            background: `${item.color}15`,
            padding: '16px',
            borderRadius: '10px',
            border: `1px solid ${item.color}40`,
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '28px', marginBottom: '8px' }}>{item.icon}</div>
            <div style={{ fontSize: '14px', color: item.color, fontWeight: '600' }}>{item.period}</div>
            <div style={{ fontSize: '11px', color: '#94a3b8' }}>{item.hijri}</div>
          </div>
        ))}
      </div>
    </SubSection>
    
    {/* 3.2 Pilgrimage Stage */}
    <SubSection title="3.2 Pilgrimage Journey Stage" color="#34d399">
      <div style={{
        background: 'rgba(30,41,59,0.6)',
        borderRadius: '12px',
        padding: '20px',
        border: '1px solid rgba(148,163,184,0.1)'
      }}>
        <div style={{ 
          display: 'flex', 
          alignItems: 'flex-start', 
          gap: '8px',
          overflowX: 'auto',
          paddingBottom: '8px'
        }}>
          {[
            { stage: 'Pre-Arrival', icon: '📱', apps: 'Nusuk, Visa apps', reviews: '"Registration confusing"' },
            { stage: 'Arrival', icon: '✈️', apps: 'Airport, Transport', reviews: '"Bus didn\'t come"' },
            { stage: 'Active Pilgrimage', icon: '🕋', apps: 'Guidance, Maps', reviews: '"App crashed during Tawaf"' },
            { stage: 'Ziyarah', icon: '🕌', apps: 'Madinah apps', reviews: '"Prayer times helpful"' },
            { stage: 'Departure', icon: '🛫', apps: 'Flight, Transport', reviews: '"Tracking useful"' }
          ].map((item, i, arr) => (
            <React.Fragment key={i}>
              <div style={{
                minWidth: '140px',
                background: 'rgba(52,211,153,0.1)',
                padding: '16px',
                borderRadius: '10px',
                border: '1px solid rgba(52,211,153,0.3)',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '32px', marginBottom: '8px' }}>{item.icon}</div>
                <div style={{ fontSize: '13px', color: '#34d399', fontWeight: '600', marginBottom: '4px' }}>
                  {item.stage}
                </div>
                <div style={{ fontSize: '10px', color: '#94a3b8', marginBottom: '8px' }}>{item.apps}</div>
                <div style={{ 
                  fontSize: '10px', 
                  color: '#cbd5e1', 
                  fontStyle: 'italic',
                  background: 'rgba(30,41,59,0.8)',
                  padding: '6px',
                  borderRadius: '4px'
                }}>
                  {item.reviews}
                </div>
              </div>
              {i < arr.length - 1 && (
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  color: '#34d399',
                  fontSize: '20px',
                  paddingTop: '40px'
                }}>→</div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
    </SubSection>
    
    {/* 3.3 Service Domain */}
    <SubSection title="3.3 Service Domain Classification" color="#34d399">
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
        {[
          { domain: 'Pilgrimage', apps: ['Nusuk', 'Tawafa', 'Eatmarna'], icon: '🕋', color: '#22d3ee' },
          { domain: 'Transportation', apps: ['Makkah Buses', 'HHR Train', 'Tanqul'], icon: '🚌', color: '#a78bfa' },
          { domain: 'Health', apps: ['Sehhaty', 'Asaafni'], icon: '🏥', color: '#34d399' },
          { domain: 'Government', apps: ['Tawakkalna', 'Absher'], icon: '🏛️', color: '#f59e0b' }
        ].map((item, i) => (
          <div key={i} style={{
            background: `${item.color}15`,
            padding: '16px',
            borderRadius: '10px',
            border: `1px solid ${item.color}40`,
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '28px', marginBottom: '8px' }}>{item.icon}</div>
            <div style={{ fontSize: '14px', color: item.color, fontWeight: '600', marginBottom: '8px' }}>{item.domain}</div>
            <div style={{ fontSize: '11px', color: '#94a3b8' }}>
              {item.apps.join(' • ')}
            </div>
          </div>
        ))}
      </div>
    </SubSection>
  </div>
);

// Step 4: Temporal-Cultural Injection
const TemporalCultural = () => (
  <div>
    <SectionTitle>Temporal-Cultural Context Injection</SectionTitle>
    <div style={{ 
      fontSize: '14px', 
      color: '#94a3b8', 
      textAlign: 'center',
      marginBottom: '24px',
      padding: '0 40px'
    }}>
      <strong style={{ color: '#22d3ee' }}>Key Insight:</strong> The same review text carries different semantic weight 
      depending on Islamic temporal context. Period tags are not just metadata—they are <em>cultural signal encoding</em>.
    </div>
    
    {/* Same Review, Different Context */}
    <SubSection title="Semantic Modulation by Period" color="#f59e0b">
      <div style={{
        background: 'rgba(30,41,59,0.6)',
        borderRadius: '12px',
        padding: '20px',
        border: '1px solid rgba(148,163,184,0.1)',
        marginBottom: '16px'
      }}>
        <div style={{ 
          textAlign: 'center', 
          padding: '16px',
          background: 'rgba(239,68,68,0.1)',
          borderRadius: '8px',
          marginBottom: '20px',
          border: '1px solid rgba(239,68,68,0.3)'
        }}>
          <div style={{ fontSize: '12px', color: '#ef4444', marginBottom: '8px' }}>SAME REVIEW TEXT</div>
          <div style={{ fontSize: '18px', color: '#e2e8f0', fontStyle: 'italic' }}>
            "The app crashed and I couldn't find my bus" ⭐☆☆☆☆
          </div>
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
          {[
            { 
              period: 'Regular Period', 
              interpretation: 'Standard Bug Report',
              severity: 'Medium',
              context: 'Normal usage, user can try again later',
              color: '#64748b',
              icon: '📆'
            },
            { 
              period: 'Hajj Season', 
              interpretation: 'CRITICAL FAILURE',
              severity: 'Critical',
              context: '2M+ pilgrims, religious obligation at stake, extreme stress',
              color: '#ef4444',
              icon: '🕋'
            },
            { 
              period: 'Ramadan', 
              interpretation: 'Time-Sensitive Failure',
              severity: 'High',
              context: 'Fasting user, Iftar timing critical, spiritual context',
              color: '#a78bfa',
              icon: '🌙'
            },
            { 
              period: 'Eid al-Adha', 
              interpretation: 'Family Disruption',
              severity: 'High',
              context: 'Family gathering, celebration travel, sacrifice rituals',
              color: '#f472b6',
              icon: '🎉'
            }
          ].map((item, i) => (
            <div key={i} style={{
              background: `${item.color}15`,
              padding: '16px',
              borderRadius: '10px',
              border: `1px solid ${item.color}40`
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                <span style={{ fontSize: '24px' }}>{item.icon}</span>
                <span style={{ fontSize: '14px', color: item.color, fontWeight: '600' }}>{item.period}</span>
              </div>
              <div style={{ 
                fontSize: '16px', 
                color: '#e2e8f0', 
                fontWeight: '700',
                marginBottom: '4px'
              }}>
                {item.interpretation}
              </div>
              <div style={{ 
                display: 'inline-block',
                fontSize: '10px', 
                color: item.color,
                background: `${item.color}30`,
                padding: '2px 8px',
                borderRadius: '4px',
                marginBottom: '8px'
              }}>
                Severity: {item.severity}
              </div>
              <div style={{ fontSize: '11px', color: '#94a3b8' }}>{item.context}</div>
            </div>
          ))}
        </div>
      </div>
    </SubSection>
    
    {/* Cultural Signal Encoding */}
    <SubSection title="Cultural Signal Encoding Function" color="#f59e0b">
      <div style={{
        background: 'rgba(30,41,59,0.8)',
        borderRadius: '12px',
        padding: '24px',
        fontFamily: 'monospace',
        fontSize: '13px',
        border: '1px solid rgba(148,163,184,0.2)'
      }}>
        <div style={{ color: '#94a3b8', marginBottom: '16px' }}>// Formalization of temporal-cultural context injection</div>
        <div style={{ color: '#f59e0b' }}>C(review, timestamp) → semantic_modifier</div>
        <div style={{ color: '#94a3b8', margin: '12px 0' }}>// Where cultural context encodes:</div>
        <div style={{ marginLeft: '20px' }}>
          {[
            { signal: '+spiritual_urgency', periods: 'Hajj, Ramadan' },
            { signal: '+crowd_stress', periods: 'Hajj, Eid, Friday' },
            { signal: '+time_sensitivity', periods: 'Ramadan (Iftar), Hajj (rituals)' },
            { signal: '+foreign_user_likelihood', periods: 'Hajj, Umrah seasons' },
            { signal: '+family_context', periods: 'Eid al-Fitr, Eid al-Adha' },
            { signal: '+service_criticality', periods: 'Active pilgrimage stages' }
          ].map((item, i) => (
            <div key={i} style={{ marginBottom: '4px' }}>
              <span style={{ color: '#34d399' }}>{item.signal}</span>
              <span style={{ color: '#64748b' }}> // {item.periods}</span>
            </div>
          ))}
        </div>
      </div>
    </SubSection>
    
    {/* Behavioral Predictions */}
    <SubSection title="Period-Based Behavioral Predictions" color="#f59e0b">
      <div style={{ overflowX: 'auto' }}>
        <table style={{ 
          width: '100%', 
          borderCollapse: 'collapse', 
          fontSize: '12px',
          background: 'rgba(30,41,59,0.6)',
          borderRadius: '12px'
        }}>
          <thead>
            <tr style={{ background: 'rgba(245,158,11,0.2)' }}>
              <th style={{ padding: '12px', textAlign: 'left', color: '#f59e0b' }}>Behavior</th>
              <th style={{ padding: '12px', textAlign: 'center', color: '#94a3b8' }}>Regular</th>
              <th style={{ padding: '12px', textAlign: 'center', color: '#94a3b8' }}>Ramadan</th>
              <th style={{ padding: '12px', textAlign: 'center', color: '#94a3b8' }}>Hajj</th>
            </tr>
          </thead>
          <tbody>
            {[
              { behavior: 'Review Time', regular: 'Daytime peak', ramadan: 'Late night (post-Iftar)', hajj: 'All hours (international)' },
              { behavior: 'Review Length', regular: 'Standard', ramadan: 'Shorter (fatigue)', hajj: 'Longer (high stakes)' },
              { behavior: 'Language Mix', regular: 'Local patterns', ramadan: 'Local patterns', hajj: 'More English (international)' },
              { behavior: 'Emotion Intensity', regular: 'Baseline', ramadan: 'Elevated', hajj: 'High urgency' }
            ].map((row, i) => (
              <tr key={i} style={{ borderTop: '1px solid rgba(148,163,184,0.1)' }}>
                <td style={{ padding: '12px', color: '#e2e8f0', fontWeight: '600' }}>{row.behavior}</td>
                <td style={{ padding: '12px', color: '#94a3b8', textAlign: 'center' }}>{row.regular}</td>
                <td style={{ padding: '12px', color: '#a78bfa', textAlign: 'center' }}>{row.ramadan}</td>
                <td style={{ padding: '12px', color: '#22d3ee', textAlign: 'center' }}>{row.hajj}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </SubSection>
  </div>
);

// Step 5: Cross-Linguistic Unification
const CrossLinguistic = () => (
  <div>
    <SectionTitle>Cross-Linguistic Unification</SectionTitle>
    <div style={{ 
      fontSize: '14px', 
      color: '#94a3b8', 
      textAlign: 'center',
      marginBottom: '24px',
      padding: '0 40px'
    }}>
      <strong style={{ color: '#22d3ee' }}>Core Innovation:</strong> Islamic temporal-cultural context is the 
      <em> shared semantic signal</em> that unifies linguistically diverse reviews across 180+ countries.
    </div>
    
    {/* Paradigm Shift */}
    <SubSection title="Paradigm Shift" color="#3b82f6">
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <div style={{
          background: 'rgba(239,68,68,0.1)',
          padding: '20px',
          borderRadius: '12px',
          border: '1px solid rgba(239,68,68,0.3)'
        }}>
          <div style={{ fontSize: '14px', color: '#ef4444', fontWeight: '700', marginBottom: '12px' }}>
            ❌ Traditional Multilingual NLP
          </div>
          <div style={{ fontSize: '13px', color: '#e2e8f0', marginBottom: '8px' }}>
            Language → determines context, culture, interpretation
          </div>
          <div style={{ fontSize: '11px', color: '#94a3b8' }}>
            Each language treated as separate semantic space
          </div>
        </div>
        <div style={{
          background: 'rgba(34,211,238,0.1)',
          padding: '20px',
          borderRadius: '12px',
          border: '1px solid rgba(34,211,238,0.3)'
        }}>
          <div style={{ fontSize: '14px', color: '#22d3ee', fontWeight: '700', marginBottom: '12px' }}>
            ✓ Islamic Service Multilingual NLP
          </div>
          <div style={{ fontSize: '13px', color: '#e2e8f0', marginBottom: '8px' }}>
            Islamic Period → provides SHARED context ACROSS languages
          </div>
          <div style={{ fontSize: '11px', color: '#94a3b8' }}>
            Language = surface variation; Religion = semantic unity
          </div>
        </div>
      </div>
    </SubSection>
    
    {/* Visual Proof */}
    <SubSection title="Cross-Linguistic Alignment Hypothesis" color="#3b82f6">
      <div style={{
        background: 'rgba(30,41,59,0.6)',
        borderRadius: '12px',
        padding: '24px',
        border: '1px solid rgba(148,163,184,0.1)'
      }}>
        <div style={{ 
          fontSize: '13px', 
          color: '#3b82f6', 
          fontWeight: '600',
          marginBottom: '16px',
          textAlign: 'center'
        }}>
          Similarity(Indonesian_Hajj, Urdu_Hajj) &gt; Similarity(Indonesian_Hajj, Indonesian_Regular)
        </div>
        
        <div style={{ display: 'flex', justifyContent: 'center', gap: '32px', flexWrap: 'wrap' }}>
          {/* Hajj Cluster */}
          <div style={{
            background: 'rgba(34,211,238,0.1)',
            padding: '20px',
            borderRadius: '50%',
            width: '200px',
            height: '200px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            border: '2px dashed rgba(34,211,238,0.5)',
            position: 'relative'
          }}>
            <div style={{ fontSize: '12px', color: '#22d3ee', fontWeight: '700', marginBottom: '8px' }}>
              🕋 HAJJ CLUSTER
            </div>
            <div style={{ fontSize: '10px', color: '#94a3b8', textAlign: 'center' }}>
              Indonesian + Urdu + Arabic + Turkish + English
            </div>
            <div style={{ fontSize: '10px', color: '#34d399', marginTop: '8px' }}>
              HIGH SIMILARITY
            </div>
          </div>
          
          {/* Regular Cluster */}
          <div style={{
            background: 'rgba(148,163,184,0.1)',
            padding: '20px',
            borderRadius: '50%',
            width: '200px',
            height: '200px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            border: '2px dashed rgba(148,163,184,0.3)'
          }}>
            <div style={{ fontSize: '12px', color: '#94a3b8', fontWeight: '700', marginBottom: '8px' }}>
              📆 REGULAR CLUSTER
            </div>
            <div style={{ fontSize: '10px', color: '#64748b', textAlign: 'center' }}>
              Same languages, different behavior patterns
            </div>
            <div style={{ fontSize: '10px', color: '#f59e0b', marginTop: '8px' }}>
              LOWER SIMILARITY TO HAJJ
            </div>
          </div>
        </div>
      </div>
    </SubSection>
    
    {/* Research Applications */}
    <SubSection title="Enabled Research Questions" color="#3b82f6">
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
        {[
          {
            question: "Universal Pain Points",
            description: "Bus app failure during Hajj generates complaints in ALL languages simultaneously",
            example: "→ Cross-linguistic clustering reveals universal issues",
            icon: "🔍"
          },
          {
            question: "Language-Specific Needs",
            description: "Arabic UI confusion only appears in non-Arabic reviews",
            example: "→ Targeted localization priorities",
            icon: "🌐"
          },
          {
            question: "Period-Aware Sentiment",
            description: "Negative sentiment during Hajj predicts app abandonment more strongly",
            example: "→ Context-weighted analysis",
            icon: "📊"
          }
        ].map((item, i) => (
          <div key={i} style={{
            background: 'rgba(59,130,246,0.1)',
            padding: '16px',
            borderRadius: '10px',
            border: '1px solid rgba(59,130,246,0.3)'
          }}>
            <div style={{ fontSize: '24px', marginBottom: '8px' }}>{item.icon}</div>
            <div style={{ fontSize: '13px', color: '#3b82f6', fontWeight: '600', marginBottom: '8px' }}>
              {item.question}
            </div>
            <div style={{ fontSize: '11px', color: '#e2e8f0', marginBottom: '8px' }}>
              {item.description}
            </div>
            <div style={{ fontSize: '10px', color: '#34d399', fontStyle: 'italic' }}>
              {item.example}
            </div>
          </div>
        ))}
      </div>
    </SubSection>
    
    {/* Final Summary */}
    <div style={{
      marginTop: '24px',
      background: 'linear-gradient(135deg, rgba(34,211,238,0.1), rgba(59,130,246,0.1), rgba(167,139,250,0.1))',
      borderRadius: '16px',
      padding: '24px',
      border: '1px solid rgba(34,211,238,0.3)',
      textAlign: 'center'
    }}>
      <div style={{ fontSize: '16px', color: '#22d3ee', fontWeight: '700', marginBottom: '12px' }}>
        🎯 Unified Contribution
      </div>
      <div style={{ fontSize: '14px', color: '#e2e8f0', maxWidth: '700px', margin: '0 auto' }}>
        A multilingual data quality taxonomy (Innovation 1) with temporal-cultural context injection 
        as cross-linguistic semantic unifier (Innovation 2) — establishing a foundation for 
        <strong style={{ color: '#34d399' }}> culturally-situated, data-centric Islamic service analytics</strong>.
      </div>
    </div>
  </div>
);

// Reusable Components
const SectionTitle = ({ children }) => (
  <h2 style={{
    fontSize: '22px',
    fontWeight: '700',
    color: '#e2e8f0',
    marginBottom: '24px',
    textAlign: 'center',
    borderBottom: '2px solid rgba(34,211,238,0.3)',
    paddingBottom: '12px'
  }}>
    {children}
  </h2>
);

const SubSection = ({ title, color, children }) => (
  <div style={{ marginBottom: '24px' }}>
    <h3 style={{
      fontSize: '16px',
      fontWeight: '600',
      color: color,
      marginBottom: '16px',
      display: 'flex',
      alignItems: 'center',
      gap: '8px'
    }}>
      <span style={{
        width: '8px',
        height: '8px',
        borderRadius: '50%',
        background: color
      }}></span>
      {title}
    </h3>
    {children}
  </div>
);

const FlowBox = ({ color, title, children }) => (
  <div style={{
    background: `${color}15`,
    border: `2px solid ${color}`,
    borderRadius: '12px',
    padding: '16px',
    textAlign: 'center'
  }}>
    <div style={{ fontSize: '14px', fontWeight: '700', color: color, marginBottom: '8px' }}>
      {title}
    </div>
    {children}
  </div>
);

const FlowArrow = () => (
  <div style={{ 
    textAlign: 'center', 
    padding: '8px',
    color: '#64748b',
    fontSize: '24px'
  }}>
    ↓
  </div>
);

const TaxonomyLevel = ({ level, title, items, color }) => (
  <div style={{
    background: `${color}15`,
    borderRadius: '8px',
    padding: '12px',
    border: `1px solid ${color}40`
  }}>
    <div style={{ fontSize: '11px', color: color, fontWeight: '700', marginBottom: '4px' }}>
      LEVEL {level}
    </div>
    <div style={{ fontSize: '13px', color: '#e2e8f0', fontWeight: '600', marginBottom: '8px' }}>
      {title}
    </div>
    <div style={{ fontSize: '10px', color: '#94a3b8' }}>
      {items.map((item, i) => (
        <div key={i}>• {item}</div>
      ))}
    </div>
  </div>
);

const InnovationBox = ({ icon, title, description }) => (
  <div style={{
    background: 'rgba(30,41,59,0.6)',
    borderRadius: '12px',
    padding: '20px',
    border: '1px solid rgba(148,163,184,0.1)'
  }}>
    <div style={{ fontSize: '32px', marginBottom: '12px' }}>{icon}</div>
    <div style={{ fontSize: '14px', color: '#22d3ee', fontWeight: '600', marginBottom: '8px' }}>
      {title}
    </div>
    <div style={{ fontSize: '12px', color: '#94a3b8' }}>
      {description}
    </div>
  </div>
);

const ExampleCard = ({ title, subtitle, examples }) => (
  <div style={{
    background: 'rgba(30,41,59,0.6)',
    borderRadius: '12px',
    padding: '16px',
    border: '1px solid rgba(148,163,184,0.1)'
  }}>
    <div style={{ fontSize: '14px', color: '#e2e8f0', fontWeight: '600', marginBottom: '4px' }}>
      {title}
    </div>
    <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '12px' }}>
      {subtitle}
    </div>
    <div style={{ display: 'grid', gap: '8px' }}>
      {examples.map((ex, i) => (
        <div key={i} style={{
          display: 'grid',
          gridTemplateColumns: '1fr auto 1fr',
          gap: '8px',
          alignItems: 'center',
          fontSize: '12px',
          padding: '8px',
          background: 'rgba(15,23,42,0.6)',
          borderRadius: '6px'
        }}>
          <div style={{ color: '#f59e0b', fontFamily: 'monospace' }}>{ex.before}</div>
          <div style={{ color: '#64748b' }}>→</div>
          <div>
            <span style={{ color: '#34d399' }}>{ex.after}</span>
            {ex.note && <span style={{ color: '#64748b', marginLeft: '8px', fontSize: '10px' }}>({ex.note})</span>}
          </div>
        </div>
      ))}
    </div>
  </div>
);

const LanguageTierCard = ({ tier, languages, accuracy, color }) => (
  <div style={{
    background: `${color}15`,
    borderRadius: '10px',
    padding: '16px',
    border: `1px solid ${color}40`,
    textAlign: 'center'
  }}>
    <div style={{ fontSize: '13px', color: color, fontWeight: '700', marginBottom: '8px' }}>
      {tier}
    </div>
    <div style={{ fontSize: '11px', color: '#e2e8f0', marginBottom: '8px' }}>
      {languages.join(', ')}
    </div>
    <div style={{ 
      fontSize: '10px', 
      color: color,
      background: `${color}30`,
      padding: '4px 8px',
      borderRadius: '4px',
      display: 'inline-block'
    }}>
      Accuracy: {accuracy}
    </div>
  </div>
);

const ScriptFamilyCard = ({ family, languages, example, color }) => (
  <div style={{
    background: `${color}10`,
    borderRadius: '10px',
    padding: '16px',
    border: `1px solid ${color}30`
  }}>
    <div style={{ fontSize: '13px', color: color, fontWeight: '600', marginBottom: '8px' }}>
      {family}
    </div>
    <div style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '8px' }}>
      {languages.join(', ')}
    </div>
    <div style={{ 
      fontSize: '14px', 
      color: '#e2e8f0',
      fontFamily: 'serif',
      letterSpacing: '1px'
    }}>
      {example}
    </div>
  </div>
);

export default TaxonomyDiagram;
